"""
HISTORICAL DATA BACKFILL
=========================
Pulls weeks/months of historical data from Binance in minutes.
Reconstructs the same snapshot format as 1_data_collector.py
so the rest of the pipeline works unchanged.

What we can get historically:
  - OHLCV klines (price, volume, taker buy ratio) — full history available
  - Funding rate history — full history available
  - Open interest history — available (5m granularity)
  - Liquidations history — NOT available via public API (live only)
  - Order book / CVD — NOT available historically (live only)

So historical training uses a subset of signals.
Live trading uses all signals.
This is fine — the historical model is still useful as a baseline.

Usage:
    python 0_backfill_history.py                  # Default: 60 days of 5m data
    python 0_backfill_history.py --days 90        # 90 days
    python 0_backfill_history.py --interval 15m   # 15-minute candles
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import argparse
from datetime import datetime, timezone, timedelta

SYMBOL   = "BTCUSDT"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

BINANCE_BASE = "https://fapi.binance.com"


# ─────────────────────────────────────────
# 1. HISTORICAL KLINES (OHLCV)
# ─────────────────────────────────────────
def fetch_klines_historical(symbol, interval, start_ms, end_ms):
    """
    Fetch all klines between start_ms and end_ms.
    Binance returns max 1500 per request — we paginate automatically.
    """
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    all_candles = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     1500,
        }
        r = requests.get(url, params=params, timeout=10)
        candles = r.json()

        if not candles or isinstance(candles, dict):
            break

        all_candles.extend(candles)

        # Next page starts after last candle's close time
        last_open_time = candles[-1][0]
        current_start = last_open_time + 1

        # If we got less than 1500, we've reached the end
        if len(candles) < 1500:
            break

        time.sleep(0.1)  # Gentle rate limiting

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)

    df["trades"] = df["trades"].astype(int)
    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# 2. HISTORICAL FUNDING RATES
# ─────────────────────────────────────────
def fetch_funding_history(symbol, start_ms, end_ms):
    """
    Funding rate is recorded every 8 hours.
    We'll forward-fill it to match kline timestamps.
    """
    url = f"{BINANCE_BASE}/fapi/v1/fundingRate"
    all_rates = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     1000,
        }
        r = requests.get(url, params=params, timeout=10)
        rates = r.json()

        if not rates or isinstance(rates, dict):
            break

        all_rates.extend(rates)

        last_time = rates[-1]["fundingTime"]
        current_start = last_time + 1

        if len(rates) < 1000:
            break

        time.sleep(0.1)

    if not all_rates:
        return pd.DataFrame()

    df = pd.DataFrame(all_rates)
    df["fundingRate"] = df["fundingRate"].astype(float)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.rename(columns={"fundingTime": "timestamp", "fundingRate": "funding_rate"})
    df = df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# 3. HISTORICAL OPEN INTEREST
# ─────────────────────────────────────────
def fetch_oi_history(symbol, interval, start_ms, end_ms):
    """
    Open interest history at 5m/15m/1h granularity.
    Available for last ~30 days only on Binance public API.
    """
    url = f"{BINANCE_BASE}/futures/data/openInterestHist"
    all_oi = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "period":    interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     500,
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        if not data or isinstance(data, dict):
            break

        all_oi.extend(data)

        last_time = data[-1]["timestamp"]
        current_start = last_time + 1

        if len(data) < 500:
            break

        time.sleep(0.1)

    if not all_oi:
        return pd.DataFrame()

    df = pd.DataFrame(all_oi)
    df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "sumOpenInterest"]].sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────
# 4. BUILD FEATURES FROM HISTORICAL DATA
# ─────────────────────────────────────────
def build_historical_features(klines, funding, oi):
    """
    Build the same feature set as 2_feature_engineering.py
    but from historical kline data instead of live snapshots.

    We lose: order book imbalance, CVD, liquidations (live only)
    We keep: price features, funding rate, open interest, volume signals
    """
    df = klines.copy()
    df = df.set_index("open_time")

    # ── PRICE FEATURES ──────────────────────
    df["f_price_return_1"]    = df["close"].pct_change(1) * 100
    df["f_price_return_3"]    = df["close"].pct_change(3) * 100
    df["f_price_return_6"]    = df["close"].pct_change(6) * 100
    df["f_price_return_12"]   = df["close"].pct_change(12) * 100
    df["f_price_volatility"]  = df["close"].pct_change().rolling(12).std() * 100
    df["f_price_vs_ma12"]     = (df["close"] / df["close"].rolling(12).mean() - 1) * 100
    df["f_price_vs_ma24"]     = (df["close"] / df["close"].rolling(24).mean() - 1) * 100
    df["f_price_vs_ma96"]     = (df["close"] / df["close"].rolling(96).mean() - 1) * 100

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["f_price_rsi"]         = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df["f_price_oversold"]    = (df["f_price_rsi"] < 30).astype(int)
    df["f_price_overbought"]  = (df["f_price_rsi"] > 70).astype(int)

    # Candle structure
    df["f_candle_direction"]  = np.where(df["close"] > df["open"], 1, -1)
    df["f_candle_size"]       = abs(df["close"] - df["open"]) / df["open"] * 100
    df["f_upper_wick"]        = (df["high"] - df[["open","close"]].max(axis=1)) / df["open"] * 100
    df["f_lower_wick"]        = (df[["open","close"]].min(axis=1) - df["low"]) / df["open"] * 100

    # ── VOLUME / TAKER FEATURES ─────────────
    df["f_volume_ma"]         = df["volume"].rolling(20).mean()
    df["f_volume_ratio"]      = df["volume"] / (df["f_volume_ma"] + 1e-10)
    df["f_volume_spike"]      = (df["f_volume_ratio"] > 2).astype(int)
    df["f_taker_buy_ratio"]   = df["taker_buy_base"] / (df["volume"] + 1e-10)
    df["f_taker_buy_delta"]   = df["f_taker_buy_ratio"].diff(1)
    df["f_taker_buy_ma"]      = df["f_taker_buy_ratio"].rolling(12).mean()
    # Persistent buying pressure
    df["f_taker_buy_vs_ma"]   = df["f_taker_buy_ratio"] - df["f_taker_buy_ma"]

    # ── FUNDING RATE ─────────────────────────
    if funding is not None and len(funding) > 0:
        # Resample funding to match kline frequency (forward fill)
        funding_indexed = funding.set_index("timestamp")["funding_rate"]
        funding_resampled = funding_indexed.reindex(df.index, method="ffill")
        df["f_fund_rate"]          = funding_resampled
        df["f_fund_extreme_long"]  = (df["f_fund_rate"] > 0.001).astype(int)
        df["f_fund_extreme_short"] = (df["f_fund_rate"] < -0.001).astype(int)
        df["f_fund_delta"]         = df["f_fund_rate"].diff(3)

    # ── OPEN INTEREST ────────────────────────
    if oi is not None and len(oi) > 0:
        oi_indexed = oi.set_index("timestamp")["sumOpenInterest"]
        oi_resampled = oi_indexed.reindex(df.index, method="ffill")
        df["f_oi"]                 = oi_resampled
        df["f_oi_change_1h"]       = df["f_oi"].pct_change(12) * 100  # 12 x 5min = 1h
        df["f_oi_change_4h"]       = df["f_oi"].pct_change(48) * 100
        df["f_oi_increasing"]      = (df["f_oi_change_1h"] > 0).astype(int)
        # OI + price divergence
        price_change_1h = df["close"].pct_change(12) * 100
        df["f_oi_price_divergence"] = price_change_1h - df["f_oi_change_1h"]

    return df.reset_index()


# ─────────────────────────────────────────
# 5. LABEL
# ─────────────────────────────────────────
def create_labels(df, forward_periods=3, threshold_pct=0.1):
    """
    Binary label: will price go UP or DOWN in next N candles?
    forward_periods=3 on 5m data = 15 minute prediction horizon
    """
    price = df["close"]
    future_price = price.shift(-forward_periods)
    price_change = (future_price - price) / price * 100

    df["label"] = np.nan
    df.loc[price_change >  threshold_pct, "label"] = 1
    df.loc[price_change < -threshold_pct, "label"] = 0

    df["label_raw_return"] = price_change

    up   = (df["label"] == 1).sum()
    down = (df["label"] == 0).sum()
    amb  = df["label"].isna().sum()
    print(f"  Labels — UP: {up} | DOWN: {down} | Ambiguous: {amb}")

    return df


# ─────────────────────────────────────────
# 6. TRAIN/TEST SPLIT (time-based)
# ─────────────────────────────────────────
def save_train_test(df, test_ratio=0.2):
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    clean = df[feature_cols + ["label", "open_time"]].dropna()

    split_idx = int(len(clean) * (1 - test_ratio))
    train = clean.iloc[:split_idx]
    test  = clean.iloc[split_idx:]

    train.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test.to_csv(f"{DATA_DIR}/test.csv",  index=False)

    print(f"  Train: {len(train)} rows  ({train['open_time'].iloc[0]} → {train['open_time'].iloc[-1]})")
    print(f"  Test:  {len(test)}  rows  ({test['open_time'].iloc[0]}  → {test['open_time'].iloc[-1]})")
    print(f"  Features: {len(feature_cols)}")

    return train, test, feature_cols


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",     type=int,   default=60,   help="Days of history to fetch")
    parser.add_argument("--interval", type=str,   default="5m", help="Candle interval: 5m, 15m, 1h")
    parser.add_argument("--symbol",   type=str,   default="BTCUSDT")
    parser.add_argument("--forward",  type=int,   default=3,    help="Forward periods for label")
    parser.add_argument("--threshold",type=float, default=0.1,  help="Min move pct to label")
    args = parser.parse_args()

    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = int((datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp() * 1000)

    print(f"=== HISTORICAL BACKFILL ===")
    print(f"Symbol:   {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Period:   last {args.days} days\n")

    # 1. Klines
    print(f"Fetching {args.days} days of {args.interval} klines...")
    klines = fetch_klines_historical(args.symbol, args.interval, start_ms, now_ms)
    print(f"  Got {len(klines)} candles  ({klines['open_time'].iloc[0]} → {klines['open_time'].iloc[-1]})")

    # 2. Funding
    print("Fetching funding rate history...")
    funding = fetch_funding_history(args.symbol, start_ms, now_ms)
    print(f"  Got {len(funding)} funding records")

    # 3. Open Interest (limited to ~30 days on public API)
    print("Fetching open interest history...")
    oi = fetch_oi_history(args.symbol, args.interval, start_ms, now_ms)
    if len(oi) == 0:
        print("  OI history not available for this range (Binance limits to ~30 days)")
        oi = None
    else:
        print(f"  Got {len(oi)} OI records")

    # 4. Build features
    print("\nBuilding features...")
    df = build_historical_features(klines, funding, oi)

    # 5. Label
    print("Creating labels...")
    df = create_labels(df, forward_periods=args.forward, threshold_pct=args.threshold)

    # 6. Save raw dataset
    raw_path = f"{DATA_DIR}/historical_{args.symbol}_{args.interval}_{args.days}d.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nRaw dataset saved → {raw_path}")

    # 7. Train/test split
    print("\nSplitting train/test...")
    train, test, feature_cols = save_train_test(df)

    print(f"\nDone. Now run: python 3_train_model.py")