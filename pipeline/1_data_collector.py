"""
CRYPTO MANIPULATION DETECTION - DATA PIPELINE
==============================================
Collects order book, liquidation, funding rate, and CVD data
from Binance (free, no API key needed for public endpoints)

Install dependencies:
    pip install requests pandas numpy websocket-client python-binance

Run:
    python 1_data_collector.py
"""

import requests
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SYMBOL = "BTCUSDT"
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

BINANCE_BASE = "https://fapi.binance.com"  # Futures API (has liquidations, funding)
COINGLASS_BASE = "https://open-api.coinglass.com/public/v2"


# ─────────────────────────────────────────
# 1. ORDER BOOK SNAPSHOT
# ─────────────────────────────────────────
def fetch_order_book(symbol=SYMBOL, limit=100):
    """
    Fetch current order book snapshot.
    Returns bid/ask walls and imbalance ratio.
    
    Imbalance > 0.6 = more buying pressure
    Imbalance < 0.4 = more selling pressure
    """
    url = f"{BINANCE_BASE}/fapi/v1/depth"
    params = {"symbol": symbol, "limit": limit}
    
    r = requests.get(url, params=params)
    data = r.json()
    
    bids = pd.DataFrame(data["bids"], columns=["price", "qty"], dtype=float)
    asks = pd.DataFrame(data["asks"], columns=["price", "qty"], dtype=float)
    
    total_bid_volume = bids["qty"].sum()
    total_ask_volume = asks["qty"].sum()
    
    # Imbalance ratio: >0.5 means more bids (buying pressure)
    imbalance = total_bid_volume / (total_bid_volume + total_ask_volume)
    
    # Find large walls (top 5% by size)
    bid_wall_threshold = bids["qty"].quantile(0.95)
    ask_wall_threshold = asks["qty"].quantile(0.95)
    
    large_bids = bids[bids["qty"] >= bid_wall_threshold]
    large_asks = asks[asks["qty"] >= ask_wall_threshold]
    
    mid_price = (bids["price"].iloc[0] + asks["price"].iloc[0]) / 2
    spread = asks["price"].iloc[0] - bids["price"].iloc[0]
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "mid_price": mid_price,
        "spread": spread,
        "spread_pct": spread / mid_price * 100,
        "bid_volume": total_bid_volume,
        "ask_volume": total_ask_volume,
        "imbalance_ratio": imbalance,           # KEY SIGNAL
        "large_bid_count": len(large_bids),
        "large_ask_count": len(large_asks),
        "largest_bid_size": bids["qty"].max(),
        "largest_ask_size": asks["qty"].max(),
        "top_bid_price": bids["price"].iloc[0],
        "top_ask_price": asks["price"].iloc[0],
    }
    
    return result, bids, asks


# ─────────────────────────────────────────
# 2. RECENT TRADES (for CVD calculation)
# ─────────────────────────────────────────
def fetch_recent_trades(symbol=SYMBOL, limit=1000):
    """
    Fetch recent trades to calculate CVD.
    CVD = Cumulative Volume Delta
    
    Buyer-initiated trades = positive delta
    Seller-initiated trades = negative delta
    
    Rising CVD with rising price = strong move
    Divergence = potential reversal
    """
    url = f"{BINANCE_BASE}/fapi/v1/aggTrades"
    params = {"symbol": symbol, "limit": limit}
    
    r = requests.get(url, params=params)
    trades = pd.DataFrame(r.json())
    
    trades["price"] = trades["p"].astype(float)
    trades["qty"] = trades["q"].astype(float)
    trades["timestamp"] = pd.to_datetime(trades["T"], unit="ms")
    trades["is_buyer_maker"] = trades["m"]  # True = seller initiated
    
    # Delta: positive if buyer aggressor, negative if seller aggressor
    trades["delta"] = np.where(trades["is_buyer_maker"], -trades["qty"], trades["qty"])
    trades["cvd"] = trades["delta"].cumsum()
    
    # Summary stats
    buy_volume = trades[~trades["is_buyer_maker"]]["qty"].sum()
    sell_volume = trades[trades["is_buyer_maker"]]["qty"].sum()
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_volume": trades["qty"].sum(),
        "buy_volume": buy_volume,
        "sell_volume": sell_volume,
        "cvd_net": trades["cvd"].iloc[-1],          # KEY SIGNAL
        "cvd_direction": "bullish" if trades["cvd"].iloc[-1] > 0 else "bearish",
        "buy_sell_ratio": buy_volume / (buy_volume + sell_volume),
        "trade_count": len(trades),
        "avg_trade_size": trades["qty"].mean(),
        "large_trade_count": len(trades[trades["qty"] > trades["qty"].quantile(0.95)]),
    }
    
    return result, trades


# ─────────────────────────────────────────
# 3. FUNDING RATE
# ─────────────────────────────────────────
def fetch_funding_rate(symbol=SYMBOL):
    """
    Funding rate = cost of holding leveraged position.
    
    High positive funding = market very long (crowded long = contrarian short signal)
    High negative funding = market very short (crowded short = contrarian long signal)
    
    Extreme funding rates often precede reversals.
    """
    url = f"{BINANCE_BASE}/fapi/v1/premiumIndex"
    params = {"symbol": symbol}
    
    r = requests.get(url, params=params)
    data = r.json()
    
    funding_rate = float(data["lastFundingRate"])
    
    # Annualized funding (funding happens every 8 hours = 3x per day = 1095x per year)
    annualized = funding_rate * 1095 * 100
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "funding_rate": funding_rate,
        "funding_rate_pct": funding_rate * 100,
        "annualized_funding_pct": annualized,       # KEY SIGNAL
        "mark_price": float(data["markPrice"]),
        "index_price": float(data["indexPrice"]),
        "sentiment": (
            "extremely_long" if funding_rate > 0.001 else
            "long" if funding_rate > 0 else
            "short" if funding_rate > -0.001 else
            "extremely_short"
        )
    }
    
    return result


# ─────────────────────────────────────────
# 4. OPEN INTEREST
# ─────────────────────────────────────────
def fetch_open_interest(symbol=SYMBOL):
    """
    Open Interest = total value of outstanding futures contracts.
    
    Rising price + rising OI = strong trend (new money entering)
    Rising price + falling OI = short squeeze (weaker move)
    Falling price + rising OI = strong downtrend
    """
    url = f"{BINANCE_BASE}/fapi/v1/openInterest"
    params = {"symbol": symbol}
    
    r = requests.get(url, params=params)
    data = r.json()
    
    # Historical OI for trend
    url_hist = f"{BINANCE_BASE}/futures/data/openInterestHist"
    params_hist = {"symbol": symbol, "period": "5m", "limit": 12}  # last hour
    r_hist = requests.get(url_hist, params=params_hist)
    hist = pd.DataFrame(r_hist.json())
    
    hist["sumOpenInterest"] = hist["sumOpenInterest"].astype(float)
    oi_change_1h = (
        (hist["sumOpenInterest"].iloc[-1] - hist["sumOpenInterest"].iloc[0])
        / hist["sumOpenInterest"].iloc[0] * 100
    ) if len(hist) > 1 else 0
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "open_interest": float(data["openInterest"]),
        "oi_change_1h_pct": oi_change_1h,           # KEY SIGNAL
        "oi_trend": "increasing" if oi_change_1h > 0 else "decreasing",
    }
    
    return result


# ─────────────────────────────────────────
# 5. LIQUIDATIONS (last 1 hour)
# ─────────────────────────────────────────
def fetch_liquidations(symbol=SYMBOL):
    """
    Liquidation clusters = forced selling/buying.
    
    Large liquidations often trigger cascades.
    Detecting the START of a liquidation cascade = strong signal.
    
    Note: Binance removed real-time liquidation feed from public API.
    We use the liquidation orders endpoint.
    """
    url = f"{BINANCE_BASE}/fapi/v1/forceOrders"
    params = {"symbol": symbol, "limit": 100}
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        
        if isinstance(data, list) and len(data) > 0:
            liq_df = pd.DataFrame(data)
            liq_df["price"] = liq_df["price"].astype(float)
            liq_df["origQty"] = liq_df["origQty"].astype(float)
            liq_df["time"] = pd.to_datetime(liq_df["time"], unit="ms")
            
            long_liqs = liq_df[liq_df["side"] == "SELL"]  # Long positions liquidated
            short_liqs = liq_df[liq_df["side"] == "BUY"]  # Short positions liquidated
            
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_liquidations": len(liq_df),
                "long_liquidations": len(long_liqs),
                "short_liquidations": len(short_liqs),
                "long_liq_volume": long_liqs["origQty"].sum(),
                "short_liq_volume": short_liqs["origQty"].sum(),
                "dominant_liquidation": "longs" if len(long_liqs) > len(short_liqs) else "shorts",
            }
        else:
            result = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_liquidations": 0,
                "note": "No recent liquidations or endpoint requires auth"
            }
    except Exception as e:
        result = {"timestamp": datetime.utcnow().isoformat(), "error": str(e)}
    
    return result


# ─────────────────────────────────────────
# 6. KLINE / OHLCV (for price context)
# ─────────────────────────────────────────
def fetch_klines(symbol=SYMBOL, interval="5m", limit=100):
    """
    Standard candlestick data.
    Used for price context and basic features.
    """
    url = f"{BINANCE_BASE}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    
    r = requests.get(url, params=params)
    data = r.json()
    
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    # Add basic features
    df["candle_direction"] = np.where(df["close"] > df["open"], 1, -1)
    df["candle_size"] = abs(df["close"] - df["open"]) / df["open"] * 100
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"] * 100
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"] * 100
    df["volume_ma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma"]  # >2 = volume spike
    df["taker_buy_ratio"] = df["taker_buy_base"] / df["volume"]  # KEY SIGNAL
    
    # Momentum
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_6"] = df["close"].pct_change(6)
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    
    return df


# ─────────────────────────────────────────
# 7. COMBINED SNAPSHOT (single data point)
# ─────────────────────────────────────────
def collect_snapshot(symbol=SYMBOL):
    """
    Collect all signals at once for a single timestamp.
    This is one row of your training data.
    """
    print(f"[{datetime.utcnow().isoformat()}] Collecting snapshot...")
    
    snapshot = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol}
    
    try:
        ob_summary, _, _ = fetch_order_book(symbol)
        snapshot.update({f"ob_{k}": v for k, v in ob_summary.items() if k != "timestamp"})
        print("  ✓ Order book")
    except Exception as e:
        print(f"  ✗ Order book: {e}")
    
    try:
        cvd_summary, _ = fetch_recent_trades(symbol)
        snapshot.update({f"cvd_{k}": v for k, v in cvd_summary.items() if k != "timestamp"})
        print("  ✓ CVD / trades")
    except Exception as e:
        print(f"  ✗ CVD: {e}")
    
    try:
        funding = fetch_funding_rate(symbol)
        snapshot.update({f"fund_{k}": v for k, v in funding.items() if k != "timestamp"})
        print("  ✓ Funding rate")
    except Exception as e:
        print(f"  ✗ Funding: {e}")
    
    try:
        oi = fetch_open_interest(symbol)
        snapshot.update({f"oi_{k}": v for k, v in oi.items() if k != "timestamp"})
        print("  ✓ Open interest")
    except Exception as e:
        print(f"  ✗ OI: {e}")
    
    try:
        liqs = fetch_liquidations(symbol)
        snapshot.update({f"liq_{k}": v for k, v in liqs.items() if k != "timestamp"})
        print("  ✓ Liquidations")
    except Exception as e:
        print(f"  ✗ Liquidations: {e}")
    
    return snapshot


# ─────────────────────────────────────────
# 8. CONTINUOUS COLLECTION LOOP
# ─────────────────────────────────────────
def run_collection_loop(interval_seconds=300, duration_hours=24):
    """
    Collect snapshots every `interval_seconds`.
    Default: every 5 minutes for 24 hours = 288 data points.
    
    For training you want at least a few weeks of data.
    Run this continuously (use screen or nohup on Linux).
    """
    total_snapshots = int(duration_hours * 3600 / interval_seconds)
    snapshots = []
    
    output_file = f"{DATA_DIR}/{SYMBOL}_snapshots_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
    
    print(f"Starting collection: {total_snapshots} snapshots over {duration_hours}h")
    print(f"Output: {output_file}\n")
    
    for i in range(total_snapshots):
        snapshot = collect_snapshot()
        snapshots.append(snapshot)
        
        # Save incrementally (don't lose data if script crashes)
        pd.DataFrame(snapshots).to_csv(output_file, index=False)
        
        print(f"  Saved snapshot {i+1}/{total_snapshots}\n")
        
        if i < total_snapshots - 1:
            time.sleep(interval_seconds)
    
    print(f"Collection complete. Data saved to {output_file}")
    return pd.DataFrame(snapshots)


# ─────────────────────────────────────────
# QUICK TEST — run single snapshot
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "loop":
        # Run continuous collection
        # Usage: python 1_data_collector.py loop
        df = run_collection_loop(interval_seconds=300, duration_hours=48)
    else:
        # Single test snapshot
        print("=== SINGLE SNAPSHOT TEST ===\n")
        
        # Test individual components
        print("1. Order Book:")
        ob, bids, asks = fetch_order_book()
        print(f"   Mid price: ${ob['mid_price']:,.2f}")
        print(f"   Imbalance: {ob['imbalance_ratio']:.3f} ({'bullish' if ob['imbalance_ratio'] > 0.5 else 'bearish'})")
        print(f"   Spread: ${ob['spread']:.2f}")
        
        print("\n2. CVD (last 1000 trades):")
        cvd, trades = fetch_recent_trades()
        print(f"   Net CVD: {cvd['cvd_net']:.2f}")
        print(f"   Direction: {cvd['cvd_direction']}")
        print(f"   Buy/Sell ratio: {cvd['buy_sell_ratio']:.3f}")
        
        print("\n3. Funding Rate:")
        fr = fetch_funding_rate()
        print(f"   Rate: {fr['funding_rate_pct']:.4f}%")
        print(f"   Annualized: {fr['annualized_funding_pct']:.1f}%")
        print(f"   Sentiment: {fr['sentiment']}")
        
        print("\n4. Open Interest:")
        oi = fetch_open_interest()
        print(f"   OI: {oi['open_interest']:,.0f} BTC")
        print(f"   1h Change: {oi['oi_change_1h_pct']:.2f}%")
        
        print("\n5. Klines (5m):")
        klines = fetch_klines(interval="5m", limit=20)
        latest = klines.iloc[-1]
        print(f"   Close: ${latest['close']:,.2f}")
        print(f"   Volume ratio: {latest['volume_ratio']:.2f}x avg")
        print(f"   Taker buy ratio: {latest['taker_buy_ratio']:.3f}")
        print(f"   RSI: {latest['rsi']:.1f}")
        
        print("\n=== Full snapshot ===")
        snapshot = collect_snapshot()
        df = pd.DataFrame([snapshot])
        print(df.T.to_string())
        
        # Save test
        df.to_csv(f"{DATA_DIR}/test_snapshot.csv", index=False)
        print(f"\nSaved to {DATA_DIR}/test_snapshot.csv")