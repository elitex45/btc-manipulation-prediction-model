"""
FEATURE ENGINEERING + LABELING
================================
Takes raw collected data and prepares it for ML model training.

Run AFTER collecting data with 1_data_collector.py

Usage:
    python 2_feature_engineering.py --input data/BTCUSDT_snapshots_*.csv
"""

import pandas as pd
import numpy as np
import os
import glob
import argparse
from datetime import datetime

DATA_DIR = "./data"


# ─────────────────────────────────────────
# LOAD AND MERGE DATA
# ─────────────────────────────────────────
def load_data(input_path=None):
    if input_path:
        files = glob.glob(input_path)
    else:
        files = glob.glob(f"{DATA_DIR}/BTCUSDT_snapshots_*.csv")
    
    if not files:
        raise FileNotFoundError(f"No data files found. Run 1_data_collector.py first.")
    
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} snapshots from {len(files)} files")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    return df


# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    """
    Build features from raw signals.
    Each feature should have an intuitive reason to predict price direction.
    """
    feat = df.copy()
    
    # ── ORDER BOOK FEATURES ──────────────────
    if "ob_imbalance_ratio" in feat.columns:
        # Raw imbalance
        feat["f_ob_imbalance"] = feat["ob_imbalance_ratio"]
        
        # Imbalance trend (is pressure building?)
        feat["f_ob_imbalance_delta"] = feat["ob_imbalance_ratio"].diff(1)
        feat["f_ob_imbalance_delta3"] = feat["ob_imbalance_ratio"].diff(3)
        
        # Extreme imbalance flags
        feat["f_ob_extreme_bid"] = (feat["ob_imbalance_ratio"] > 0.65).astype(int)
        feat["f_ob_extreme_ask"] = (feat["ob_imbalance_ratio"] < 0.35).astype(int)
        
        # Wall ratio (spoofing detection proxy)
        if "ob_large_bid_count" in feat.columns:
            feat["f_ob_wall_ratio"] = (
                feat["ob_large_bid_count"] / (feat["ob_large_bid_count"] + feat["ob_large_ask_count"] + 1)
            )
    
    # ── CVD FEATURES ────────────────────────
    if "cvd_cvd_net" in feat.columns:
        feat["f_cvd_net"] = feat["cvd_cvd_net"]
        feat["f_cvd_delta"] = feat["cvd_cvd_net"].diff(1)
        feat["f_cvd_delta3"] = feat["cvd_cvd_net"].diff(3)
        
        # CVD direction consistency
        feat["f_cvd_bullish"] = (feat["cvd_cvd_net"] > 0).astype(int)
        
        # Buy/sell ratio
        if "cvd_buy_sell_ratio" in feat.columns:
            feat["f_cvd_buy_ratio"] = feat["cvd_buy_sell_ratio"]
            feat["f_cvd_buy_ratio_delta"] = feat["cvd_buy_sell_ratio"].diff(1)
        
        # Large trade activity
        if "cvd_large_trade_count" in feat.columns:
            feat["f_cvd_large_trades"] = feat["cvd_large_trade_count"]
            feat["f_cvd_large_trades_ma"] = feat["cvd_large_trade_count"].rolling(6).mean()
            feat["f_cvd_large_trades_spike"] = (
                feat["cvd_large_trade_count"] > feat["f_cvd_large_trades_ma"] * 2
            ).astype(int)
    
    # ── FUNDING RATE FEATURES ────────────────
    if "fund_funding_rate" in feat.columns:
        feat["f_fund_rate"] = feat["fund_funding_rate"]
        feat["f_fund_extreme_long"] = (feat["fund_funding_rate"] > 0.001).astype(int)
        feat["f_fund_extreme_short"] = (feat["fund_funding_rate"] < -0.001).astype(int)
        feat["f_fund_delta"] = feat["fund_funding_rate"].diff(3)
    
    # ── OPEN INTEREST FEATURES ───────────────
    if "oi_open_interest" in feat.columns:
        feat["f_oi_change"] = feat["oi_oi_change_1h_pct"]
        feat["f_oi_increasing"] = (feat["oi_oi_change_1h_pct"] > 0).astype(int)
        
        # OI + price divergence (if price rises but OI falls = short squeeze)
        if "ob_mid_price" in feat.columns:
            price_change = feat["ob_mid_price"].pct_change(6) * 100
            oi_change = feat["oi_oi_change_1h_pct"]
            feat["f_oi_price_divergence"] = price_change - oi_change
    
    # ── LIQUIDATION FEATURES ─────────────────
    if "liq_total_liquidations" in feat.columns:
        feat["f_liq_total"] = feat["liq_total_liquidations"]
        feat["f_liq_total_ma"] = feat["liq_total_liquidations"].rolling(6).mean()
        feat["f_liq_spike"] = (
            feat["liq_total_liquidations"] > feat["f_liq_total_ma"] * 3
        ).astype(int)
        
        if "liq_long_liquidations" in feat.columns:
            feat["f_liq_long_ratio"] = (
                feat["liq_long_liquidations"] / (feat["liq_total_liquidations"] + 1)
            )
    
    # ── PRICE FEATURES ──────────────────────
    if "ob_mid_price" in feat.columns:
        price = feat["ob_mid_price"]
        
        feat["f_price_return_1"] = price.pct_change(1) * 100
        feat["f_price_return_3"] = price.pct_change(3) * 100
        feat["f_price_return_6"] = price.pct_change(6) * 100
        
        # Volatility (rolling std of returns)
        feat["f_price_volatility"] = price.pct_change().rolling(12).std() * 100
        
        # Price vs moving averages
        feat["f_price_vs_ma12"] = (price / price.rolling(12).mean() - 1) * 100
        feat["f_price_vs_ma24"] = (price / price.rolling(24).mean() - 1) * 100
        
        # RSI-like momentum
        delta = price.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        feat["f_price_rsi"] = 100 - (100 / (1 + rs))
        feat["f_price_oversold"] = (feat["f_price_rsi"] < 30).astype(int)
        feat["f_price_overbought"] = (feat["f_price_rsi"] > 70).astype(int)
    
    # ── COMPOSITE / CONFLUENCE SIGNALS ───────
    # These combine multiple signals — strong confluence = stronger prediction
    
    if "f_ob_imbalance" in feat.columns and "f_cvd_bullish" in feat.columns:
        # Bullish confluence: OB shows bid pressure AND CVD bullish
        feat["f_bull_confluence"] = (
            (feat["f_ob_imbalance"] > 0.55) & 
            (feat["f_cvd_bullish"] == 1)
        ).astype(int)
        
        feat["f_bear_confluence"] = (
            (feat["f_ob_imbalance"] < 0.45) & 
            (feat["f_cvd_bullish"] == 0)
        ).astype(int)
    
    return feat


# ─────────────────────────────────────────
# LABELING
# ─────────────────────────────────────────
def create_labels(df, forward_periods=3, threshold_pct=0.1):
    """
    Create binary labels: will price go UP or DOWN?
    
    forward_periods: how many snapshots ahead to look
                    (3 snapshots × 5min = 15min prediction horizon)
    threshold_pct: minimum move to count as signal
                   (0.1 = only label if price moves >0.1%)
    
    Labels:
        1 = price goes UP by threshold_pct in next N periods
        0 = price goes DOWN by threshold_pct in next N periods
        NaN = no clear direction (filtered out during training)
    """
    if "ob_mid_price" not in df.columns:
        raise ValueError("Need ob_mid_price column for labeling")
    
    price = df["ob_mid_price"]
    future_price = price.shift(-forward_periods)
    
    price_change_pct = (future_price - price) / price * 100
    
    # Binary label with threshold
    df["label"] = np.nan
    df.loc[price_change_pct > threshold_pct, "label"] = 1    # UP
    df.loc[price_change_pct < -threshold_pct, "label"] = 0   # DOWN
    # NaN = ambiguous (within threshold) — we'll filter these out
    
    # Also save raw return for analysis
    df["label_raw_return"] = price_change_pct
    df["label_forward_periods"] = forward_periods
    
    up_count = (df["label"] == 1).sum()
    down_count = (df["label"] == 0).sum()
    ambiguous = df["label"].isna().sum()
    
    print(f"\nLabel distribution:")
    print(f"  UP:        {up_count} ({up_count/len(df)*100:.1f}%)")
    print(f"  DOWN:      {down_count} ({down_count/len(df)*100:.1f}%)")
    print(f"  Ambiguous: {ambiguous} ({ambiguous/len(df)*100:.1f}%) — will be filtered")
    
    return df


# ─────────────────────────────────────────
# SELECT FINAL FEATURE COLUMNS
# ─────────────────────────────────────────
def get_feature_columns(df):
    """Return only engineered feature columns (start with 'f_')"""
    return [col for col in df.columns if col.startswith("f_")]


# ─────────────────────────────────────────
# PREPARE TRAIN/TEST SPLIT
# ─────────────────────────────────────────
def prepare_train_test(df, test_ratio=0.2):
    """
    CRITICAL: Use TIME-BASED split, NOT random split.
    
    Random split = LOOKAHEAD BIAS = your model will appear to work
    but fail in production because it trained on future data.
    
    Always use the LAST X% of time as test set.
    """
    feature_cols = get_feature_columns(df)
    
    # Filter out rows with NaN labels or missing features
    clean = df[feature_cols + ["label", "timestamp"]].dropna()
    
    # TIME-BASED SPLIT
    split_idx = int(len(clean) * (1 - test_ratio))
    
    train = clean.iloc[:split_idx]
    test = clean.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train["label"]
    X_test = test[feature_cols]
    y_test = test["label"]
    
    print(f"\nTrain/Test split (time-based):")
    print(f"  Train: {len(train)} samples ({train['timestamp'].min()} → {train['timestamp'].max()})")
    print(f"  Test:  {len(test)} samples ({test['timestamp'].min()} → {test['timestamp'].max()})")
    print(f"  Features: {len(feature_cols)}")
    
    return X_train, y_train, X_test, y_test, feature_cols


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Glob pattern for input CSV files")
    parser.add_argument("--forward", type=int, default=3, help="Forward periods for labeling")
    parser.add_argument("--threshold", type=float, default=0.1, help="Min price move pct to label")
    args = parser.parse_args()
    
    print("=== FEATURE ENGINEERING ===\n")
    
    # Load
    df = load_data(args.input)
    
    # Engineer features
    print("\nEngineering features...")
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Created {len(feature_cols)} features: {feature_cols}")
    
    # Label
    print(f"\nLabeling (forward={args.forward} periods, threshold={args.threshold}%)...")
    df = create_labels(df, forward_periods=args.forward, threshold_pct=args.threshold)
    
    # Split
    X_train, y_train, X_test, y_test, feature_cols = prepare_train_test(df)
    
    # Save prepared datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test_df.to_csv(f"{DATA_DIR}/test.csv", index=False)
    
    print(f"\nSaved:")
    print(f"  {DATA_DIR}/train.csv")
    print(f"  {DATA_DIR}/test.csv")
    
    # Feature stats
    print("\nFeature correlations with label (top 10):")
    corr = train_df.corr()["label"].drop("label").abs().sort_values(ascending=False)
    print(corr.head(10).to_string())