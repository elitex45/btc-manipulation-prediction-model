"""
TARDIS DATA PROCESSOR
======================
Reads downloaded Tardis files and converts them into the same
feature format as our existing pipeline, but with the full
manipulation signal set:

  From book_snapshot_25:  order book imbalance, bid/ask walls, spread
  From trades:            CVD, large trade detection, buy/sell ratio
  From liquidations:      liquidation cascade signals
  From derivative_ticker: funding rate, open interest at tick level

Then resamples everything to 5-minute bars and merges with
OHLCV klines from Binance, producing a rich training dataset.

Usage:
    python 0b_process_tardis.py                 # Process all downloaded months
    python 0b_process_tardis.py --month 2024-01 # Single month
    python 0b_process_tardis.py --interval 15m  # 15min bars instead of 5m
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings("ignore")

TARDIS_DIR = "./data/tardis"
DATA_DIR   = "./data"
SYMBOL     = "BTCUSDT"


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def parse_timestamp(series):
    """Handle both millisecond and nanosecond timestamps from Tardis."""
    ts = pd.to_numeric(series, errors="coerce")
    # Tardis uses microseconds
    try:
        result = pd.to_datetime(ts, unit="us", utc=True)
    except Exception:
        result = pd.to_datetime(ts, unit="ms", utc=True)
    return result


def available_months(data_type):
    """List months that have been downloaded for a given data type."""
    path = os.path.join(TARDIS_DIR, data_type)
    if not os.path.exists(path):
        return []
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    months = [os.path.basename(f)[:7] for f in files]
    return months


# ─────────────────────────────────────────
# 1. PROCESS ORDER BOOK SNAPSHOTS
# ─────────────────────────────────────────
def process_book_snapshots(filepath, freq="5min"):
    """
    Tardis book_snapshot_25 columns:
      timestamp, exchange, symbol,
      bids[0].price, bids[0].amount, ... bids[24].price, bids[24].amount
      asks[0].price, asks[0].amount, ... asks[24].price, asks[24].amount

    We compute per-snapshot:
      - imbalance ratio (bid vol / total vol)
      - large wall detection (top 5% sizes)
      - spread
      - bid/ask depth at 0.1%, 0.5%, 1% from mid
    Then resample to freq bars (take mean/last).
    """
    print(f"  Processing order book: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False)

    df["ts"] = parse_timestamp(df["timestamp"])
    df = df.dropna(subset=["ts"]).sort_values("ts")

    # Extract bid/ask columns
    bid_price_cols  = [c for c in df.columns if c.startswith("bids[") and c.endswith(".price")]
    bid_amount_cols = [c for c in df.columns if c.startswith("bids[") and c.endswith(".amount")]
    ask_price_cols  = [c for c in df.columns if c.startswith("asks[") and c.endswith(".price")]
    ask_amount_cols = [c for c in df.columns if c.startswith("asks[") and c.endswith(".amount")]

    for cols in [bid_price_cols, bid_amount_cols, ask_price_cols, ask_amount_cols]:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

    bid_amounts = df[bid_amount_cols].values
    ask_amounts = df[ask_amount_cols].values
    bid_prices  = df[bid_price_cols].values
    ask_prices  = df[ask_price_cols].values

    total_bid = np.nansum(bid_amounts, axis=1)
    total_ask = np.nansum(ask_amounts, axis=1)

    best_bid = bid_prices[:, 0]
    best_ask = ask_prices[:, 0]
    mid      = (best_bid + best_ask) / 2.0
    spread   = best_ask - best_bid

    imbalance = total_bid / (total_bid + total_ask + 1e-10)

    # Wall detection: is any single level > 3x average size?
    avg_bid_size = np.nanmean(bid_amounts, axis=1)
    avg_ask_size = np.nanmean(ask_amounts, axis=1)
    max_bid_size = np.nanmax(bid_amounts, axis=1)
    max_ask_size = np.nanmax(ask_amounts, axis=1)
    bid_wall = (max_bid_size > avg_bid_size * 3).astype(float)
    ask_wall = (max_ask_size > avg_ask_size * 3).astype(float)

    result = pd.DataFrame({
        "ts":                df["ts"].values,
        "ob_mid_price":      mid,
        "ob_spread":         spread,
        "ob_spread_bps":     spread / (mid + 1e-10) * 10000,
        "ob_imbalance":      imbalance,
        "ob_bid_volume":     total_bid,
        "ob_ask_volume":     total_ask,
        "ob_bid_wall":       bid_wall,
        "ob_ask_wall":       ask_wall,
        "ob_bid_wall_size":  max_bid_size,
        "ob_ask_wall_size":  max_ask_size,
    })

    result = result.set_index("ts")
    resampled = result.resample(freq).agg({
        "ob_mid_price":     "last",
        "ob_spread":        "mean",
        "ob_spread_bps":    "mean",
        "ob_imbalance":     "mean",
        "ob_bid_volume":    "mean",
        "ob_ask_volume":    "mean",
        "ob_bid_wall":      "max",
        "ob_ask_wall":      "max",
        "ob_bid_wall_size": "max",
        "ob_ask_wall_size": "max",
    })

    print(f"    Rows: {len(df):,} snapshots → {len(resampled):,} {freq} bars")
    return resampled


# ─────────────────────────────────────────
# 2. PROCESS TRADES (CVD)
# ─────────────────────────────────────────
def process_trades(filepath, freq="5min"):
    """
    Tardis trades columns:
      timestamp, exchange, symbol, id, side, price, amount

    side: 'buy' = buyer aggressor (positive delta)
          'sell' = seller aggressor (negative delta)

    CVD = cumulative sum of signed volume.
    """
    print(f"  Processing trades (CVD): {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False)

    df["ts"]     = parse_timestamp(df["timestamp"])
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["ts", "price", "amount"]).sort_values("ts")

    # Signed volume
    df["delta"] = np.where(df["side"] == "buy", df["amount"], -df["amount"])
    df["is_buy"] = (df["side"] == "buy").astype(float)
    df["dollar_volume"] = df["price"] * df["amount"]

    # Pre-compute buy volume and dollar-weighted price before resampling
    df["buy_vol"]     = df["amount"].where(df["side"] == "buy", 0)
    df["dollar_vol"]  = df["price"] * df["amount"]
    df["price_x_amt"] = df["price"] * df["amount"]  # for VWAP numerator

    df = df.set_index("ts")
    resampled = df.resample(freq).agg(
        cvd_net         = ("delta",       "sum"),
        cvd_buy_vol     = ("buy_vol",     "sum"),
        cvd_total_vol   = ("amount",      "sum"),
        cvd_trade_count = ("amount",      "count"),
        cvd_dollar_vol  = ("dollar_vol",  "sum"),
        cvd_avg_size    = ("amount",      "mean"),
        cvd_max_size    = ("amount",      "max"),
        cvd_vwap_num    = ("price_x_amt", "sum"),
    )

    resampled["cvd_buy_ratio"]    = resampled["cvd_buy_vol"] / (resampled["cvd_total_vol"] + 1e-10)
    resampled["cvd_large_trades"] = resampled["cvd_max_size"] / (resampled["cvd_avg_size"] + 1e-10)
    resampled["cvd_cumulative"]   = resampled["cvd_net"].cumsum()
    resampled["cvd_vwap"]         = resampled["cvd_vwap_num"] / (resampled["cvd_dollar_vol"] + 1e-10)
    resampled = resampled.drop(columns=["cvd_vwap_num"])

    print(f"    Rows: {len(df):,} trades → {len(resampled):,} {freq} bars")
    return resampled


# ─────────────────────────────────────────
# 3. PROCESS LIQUIDATIONS
# ─────────────────────────────────────────
def process_liquidations(filepath, freq="5min"):
    """
    Tardis liquidations columns:
      timestamp, exchange, symbol, id, side, price, amount

    side: 'sell' = long position liquidated (forced selling)
          'buy'  = short position liquidated (forced buying)
    """
    print(f"  Processing liquidations: {os.path.basename(filepath)}")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except pd.errors.EmptyDataError:
        print(f"    Empty file — no liquidations this day")
        return pd.DataFrame()

    if len(df) == 0:
        print("    Empty file — no liquidations this day")
        return pd.DataFrame()

    # Filter for our symbol
    if "symbol" in df.columns:
        df = df[df["symbol"].str.contains(SYMBOL.replace("USDT", ""), na=False)]

    df["ts"]     = parse_timestamp(df["timestamp"])
    df["price"]  = pd.to_numeric(df["price"],  errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")

    if len(df) == 0:
        return pd.DataFrame()

    df["is_long_liq"]  = (df["side"] == "sell").astype(float)  # Long liquidated
    df["is_short_liq"] = (df["side"] == "buy").astype(float)   # Short liquidated
    df["liq_value"]    = df["price"] * df["amount"]

    df = df.set_index("ts")
    resampled = df.resample(freq).agg(
        liq_total_count  = ("amount",       "count"),
        liq_long_count   = ("is_long_liq",  "sum"),
        liq_short_count  = ("is_short_liq", "sum"),
        liq_total_value  = ("liq_value",    "sum"),
        liq_max_value    = ("liq_value",    "max"),
    ).fillna(0)

    resampled["liq_long_ratio"]  = resampled["liq_long_count"]  / (resampled["liq_total_count"] + 1e-10)
    resampled["liq_cascade"]     = (resampled["liq_total_count"] > resampled["liq_total_count"].rolling(12).mean() * 3).astype(float)

    print(f"    Rows: {len(df):,} liquidations → {len(resampled):,} {freq} bars (most will be 0)")
    return resampled


# ─────────────────────────────────────────
# 4. PROCESS DERIVATIVE TICKER
# ─────────────────────────────────────────
def process_derivative_ticker(filepath, freq="5min"):
    """
    Tardis derivative_ticker columns:
      timestamp, exchange, symbol,
      lastPrice, openInterest, fundingRate, indexPrice, markPrice,
      nextFundingTime, predictedFundingRate
    """
    print(f"  Processing derivative ticker: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, low_memory=False)

    df["ts"] = parse_timestamp(df["timestamp"])
    for col in ["openInterest", "fundingRate", "lastPrice", "markPrice", "indexPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["ts"]).sort_values("ts")
    df = df.set_index("ts")

    agg_dict = {}
    if "openInterest" in df.columns:
        agg_dict["deriv_open_interest"] = ("openInterest", "last")
    if "fundingRate" in df.columns:
        agg_dict["deriv_funding_rate"]  = ("fundingRate",  "last")
    if "lastPrice" in df.columns:
        agg_dict["deriv_last_price"]    = ("lastPrice",    "last")
    if "markPrice" in df.columns:
        agg_dict["deriv_mark_price"]    = ("markPrice",    "last")

    if not agg_dict:
        print(f"    [!] No usable columns in derivative ticker file")
        return pd.DataFrame()

    print(f"    Columns found: {list(agg_dict.keys())}")
    resampled = df.resample(freq).agg(**agg_dict)

    if "deriv_open_interest" in resampled.columns:
        resampled["deriv_oi_change_1h"]  = resampled["deriv_open_interest"].pct_change(12) * 100
        resampled["deriv_oi_change_4h"]  = resampled["deriv_open_interest"].pct_change(48) * 100

    if "deriv_funding_rate" in resampled.columns:
        resampled["deriv_fund_extreme_long"]  = (resampled["deriv_funding_rate"] > 0.001).astype(float)
        resampled["deriv_fund_extreme_short"] = (resampled["deriv_funding_rate"] < -0.001).astype(float)
        resampled["deriv_fund_delta"]         = resampled["deriv_funding_rate"].diff(3)

    print(f"    Rows: {len(df):,} ticks → {len(resampled):,} {freq} bars")
    return resampled


# ─────────────────────────────────────────
# 5. BUILD FEATURES FROM MERGED DATA
# ─────────────────────────────────────────
def build_features(merged):
    """
    Build ML features (all prefixed f_) from the merged dataframe.
    """
    df = merged.copy()

    # ── ORDER BOOK ───────────────────────
    if "ob_imbalance" in df.columns:
        df["f_ob_imbalance"]        = df["ob_imbalance"]
        df["f_ob_imbalance_delta"]  = df["ob_imbalance"].diff(1)
        df["f_ob_imbalance_delta3"] = df["ob_imbalance"].diff(3)
        df["f_ob_extreme_bid"]      = (df["ob_imbalance"] > 0.65).astype(float)
        df["f_ob_extreme_ask"]      = (df["ob_imbalance"] < 0.35).astype(float)
        df["f_ob_spread_bps"]       = df.get("ob_spread_bps", np.nan)
        df["f_ob_bid_wall"]         = df.get("ob_bid_wall", 0)
        df["f_ob_ask_wall"]         = df.get("ob_ask_wall", 0)
        df["f_ob_wall_imbalance"]   = df.get("ob_bid_wall", 0) - df.get("ob_ask_wall", 0)

    # ── CVD ──────────────────────────────
    if "cvd_net" in df.columns:
        df["f_cvd_net"]           = df["cvd_net"]
        df["f_cvd_net_delta"]     = df["cvd_net"].diff(1)
        df["f_cvd_buy_ratio"]     = df.get("cvd_buy_ratio", np.nan)
        df["f_cvd_buy_ratio_d1"]  = df["f_cvd_buy_ratio"].diff(1) if "f_cvd_buy_ratio" in df else np.nan
        df["f_cvd_cumulative"]    = df.get("cvd_cumulative", np.nan)
        df["f_cvd_large_trades"]  = df.get("cvd_large_trades", np.nan)
        df["f_cvd_bullish"]       = (df["cvd_net"] > 0).astype(float)
        df["f_cvd_dollar_vol"]    = df.get("cvd_dollar_vol", np.nan)

    # ── LIQUIDATIONS ─────────────────────
    if "liq_total_count" in df.columns:
        df["f_liq_count"]         = df["liq_total_count"]
        df["f_liq_long_ratio"]    = df.get("liq_long_ratio", np.nan)
        df["f_liq_value"]         = df.get("liq_total_value", np.nan)
        df["f_liq_cascade"]       = df.get("liq_cascade", 0)
        liq_ma = df["liq_total_count"].rolling(12).mean()
        df["f_liq_spike"]         = (df["liq_total_count"] > liq_ma * 3).astype(float)

    # ── FUNDING / OI ─────────────────────
    if "deriv_funding_rate" in df.columns:
        df["f_fund_rate"]          = df["deriv_funding_rate"]
        df["f_fund_extreme_long"]  = df.get("deriv_fund_extreme_long", 0)
        df["f_fund_extreme_short"] = df.get("deriv_fund_extreme_short", 0)
        df["f_fund_delta"]         = df.get("deriv_fund_delta", np.nan)

    if "deriv_open_interest" in df.columns:
        df["f_oi"]                 = df["deriv_open_interest"]
        df["f_oi_change_1h"]       = df.get("deriv_oi_change_1h", np.nan)
        df["f_oi_change_4h"]       = df.get("deriv_oi_change_4h", np.nan)
        df["f_oi_increasing"]      = (df.get("deriv_oi_change_1h", 0) > 0).astype(float)

    # ── PRICE (from ob_mid_price or close) ──
    price_col = "ob_mid_price" if "ob_mid_price" in df.columns else "close"
    if price_col in df.columns:
        p = df[price_col]
        df["f_price_return_1"]   = p.pct_change(1) * 100
        df["f_price_return_3"]   = p.pct_change(3) * 100
        df["f_price_return_6"]   = p.pct_change(6) * 100
        df["f_price_return_12"]  = p.pct_change(12) * 100
        df["f_price_volatility"] = p.pct_change().rolling(12).std() * 100
        df["f_price_vs_ma12"]    = (p / p.rolling(12).mean() - 1) * 100
        df["f_price_vs_ma24"]    = (p / p.rolling(24).mean() - 1) * 100
        df["f_price_vs_ma96"]    = (p / p.rolling(96).mean() - 1) * 100
        delta = p.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        df["f_price_rsi"]        = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df["f_price_overbought"] = (df["f_price_rsi"] > 70).astype(float)
        df["f_price_oversold"]   = (df["f_price_rsi"] < 30).astype(float)

    # ── CONFLUENCE ───────────────────────
    if "f_ob_imbalance" in df.columns and "f_cvd_bullish" in df.columns:
        df["f_bull_confluence"] = (
            (df["f_ob_imbalance"] > 0.55) &
            (df["f_cvd_bullish"] == 1)
        ).astype(float)
        df["f_bear_confluence"] = (
            (df["f_ob_imbalance"] < 0.45) &
            (df["f_cvd_bullish"] == 0)
        ).astype(float)

    return df


# ─────────────────────────────────────────
# 6. LABEL
# ─────────────────────────────────────────
def create_labels(df, price_col, forward_periods=3, threshold_pct=0.1):
    price        = df[price_col]
    future_price = price.shift(-forward_periods)
    change       = (future_price - price) / price * 100

    df["label"]            = np.nan
    df.loc[change >  threshold_pct, "label"] = 1.0
    df.loc[change < -threshold_pct, "label"] = 0.0
    df["label_raw_return"] = change
    return df


# ─────────────────────────────────────────
# 7. PROCESS ONE MONTH
# ─────────────────────────────────────────
def process_month(year, month, freq="5min"):
    """Load all available data types for one month, merge, and return features."""
    month_str = f"{year}-{month:02d}-01"
    frames    = {}

    # Book snapshots
    path = os.path.join(TARDIS_DIR, "book_snapshot_25", f"{year}-{month:02d}-01.csv")
    if os.path.exists(path):
        frames["book"] = process_book_snapshots(path, freq)

    # Trades
    path = os.path.join(TARDIS_DIR, "trades", f"{year}-{month:02d}-01.csv")
    if os.path.exists(path):
        frames["trades"] = process_trades(path, freq)

    # Liquidations
    path = os.path.join(TARDIS_DIR, "liquidations", f"{year}-{month:02d}-01.csv")
    if os.path.exists(path):
        liq = process_liquidations(path, freq)
        if len(liq) > 0:
            frames["liq"] = liq

    # Derivative ticker
    path = os.path.join(TARDIS_DIR, "derivative_ticker", f"{year}-{month:02d}-01.csv")
    if os.path.exists(path):
        frames["deriv"] = process_derivative_ticker(path, freq)

    if not frames:
        print(f"  No data found for {month_str}")
        return None

    # Normalize all indexes to UTC-aware DatetimeIndex before merging
    for name in list(frames.keys()):
        df = frames[name]
        idx = df.index
        # Skip if not a DatetimeIndex (e.g. RangeIndex from failed resample)
        if not isinstance(idx, pd.DatetimeIndex):
            try:
                frames[name].index = pd.to_datetime(idx, utc=True)
            except Exception:
                print(f"    [!] Dropping {name} — could not convert index to DatetimeIndex")
                del frames[name]
                continue
            idx = frames[name].index
        if idx.tz is None:
            frames[name].index = idx.tz_localize("UTC")
        else:
            frames[name].index = idx.tz_convert("UTC")

    # Merge all on timestamp index
    merged = None
    for name, df in frames.items():
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="outer")

    merged = merged.sort_index()

    # Build features
    merged = build_features(merged)

    # Label (use ob_mid_price if available, else fall back)
    price_col = "ob_mid_price" if "ob_mid_price" in merged.columns else None
    if price_col is None:
        print(f"  No price column found for {month_str}, skipping labels")
        return merged

    merged = create_labels(merged, price_col)

    feature_cols = [c for c in merged.columns if c.startswith("f_")]
    print(f"  Month {month_str}: {len(merged):,} bars, {len(feature_cols)} features")

    return merged


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month",    type=str, default=None, help="Single month YYYY-MM")
    parser.add_argument("--interval", type=str, default="5min", help="Bar interval: 5min, 15min, 1h")
    parser.add_argument("--forward",  type=int, default=3,    help="Forward periods for label")
    parser.add_argument("--threshold",type=float, default=0.1, help="Min move pct to label")
    args = parser.parse_args()

    print("=== TARDIS DATA PROCESSOR ===\n")

    # Find available months (intersection of what's downloaded)
    book_months = set(available_months("book_snapshot_25"))
    trade_months = set(available_months("trades"))
    all_months = sorted(book_months | trade_months |
                        set(available_months("liquidations")) |
                        set(available_months("derivative_ticker")))

    if not all_months:
        print("No Tardis data found. Run 0a_tardis_download.py first.")
        exit(1)

    if args.month:
        all_months = [m for m in all_months if m.startswith(args.month)]

    print(f"Processing {len(all_months)} months: {all_months[0]} → {all_months[-1]}\n")

    all_frames = []
    for month_str in all_months:
        year, month = int(month_str[:4]), int(month_str[5:7])
        print(f"\n[{month_str}]")
        df = process_month(year, month, freq=args.interval)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        print("No data processed.")
        exit(1)

    # Combine all months
    combined = pd.concat(all_frames, axis=0)
    combined = combined.sort_index()

    feature_cols = [c for c in combined.columns if c.startswith("f_")]
    clean = combined[feature_cols + ["label"]].dropna(subset=["label"])

    print(f"\n{'='*60}")
    print(f"Total bars:     {len(combined):,}")
    print(f"Labeled bars:   {len(clean):,}")
    print(f"Features:       {len(feature_cols)}")

    up   = (clean["label"] == 1).sum()
    down = (clean["label"] == 0).sum()
    print(f"Label split:    UP={up:,} ({up/len(clean)*100:.1f}%) | DOWN={down:,} ({down/len(clean)*100:.1f}%)")

    # Time-based train/test split
    split_idx = int(len(clean) * 0.8)
    train = clean.iloc[:split_idx]
    test  = clean.iloc[split_idx:]

    train.to_csv(f"{DATA_DIR}/train.csv", index=True)
    test.to_csv(f"{DATA_DIR}/test.csv",  index=True)

    print(f"\nTrain: {len(train):,} rows → {DATA_DIR}/train.csv")
    print(f"Test:  {len(test):,}  rows → {DATA_DIR}/test.csv")
    print(f"\nNext: python 3_train_model.py")