# Polymarket BTC Mispricing Detector

## What We're Building and Why

### The Starting Point

The goal was to build an AI model to predict markets on Polymarket — specifically the short-term BTC up/down markets (5m, 15m windows).

The first instinct was to predict price movements directly. But that runs into a hard wall immediately: institutions have entered crypto. They have co-located servers, order flow data, and capital that dwarfs any individual. Trying to predict *normal* price action against them is a losing game.

The reframe that changed everything: **manipulation itself has patterns**.

Whales don't act randomly. They have objectives — liquidating overleveraged positions, triggering stop hunts, accumulating at specific levels — and achieving those objectives leaves fingerprints in the data. Order book walls appear and disappear. Funding rates get pushed to extremes. Liquidation cascades trigger in sequence. These are detectable signals.

So the goal shifted from "predict price" to "detect what the manipulator is doing early enough to follow it."

### Why Binance and Not Polymarket Directly

Polymarket is downstream of Binance. The yes/no prices on Polymarket update *because* BTC moved on Binance, not the other way around. Polymarket's liquidity is thin, spreads are wide, and it is a reflection — not a source — of price discovery.

Training on Polymarket data alone would mean training on a lagging, noisy echo of the real signal.

The actual edge comes from the gap between:
- What Binance data + our model says the probability *should* be
- What Polymarket is currently *pricing* that probability at

That gap, when it's large enough and confirmed by our directional signal, is the bet.

### The Causal Chain

```
Whale/institution acts
        ↓
Fingerprints appear in Binance data
(order book walls, CVD divergence, funding extremes, liquidation cascades)
        ↓
Our ML model detects directional signal
        ↓
Log-normal model computes fair probability
(given current BTC price, target price, time to expiry)
        ↓
Compare against Polymarket's current yes/no price
        ↓
If gap > 4%  →  Edge exists  →  Kelly-sized bet
```

---

## The Signals We Use

These are not arbitrary technical indicators. Each one has a specific reason to predict short-term direction.

| Signal | Source | What It Detects |
|--------|--------|----------------|
| **Order Book Imbalance** | Binance order book | Ratio of bid vs ask volume. >0.6 = strong buy pressure. Walls appearing/disappearing = spoofing. |
| **CVD (Cumulative Volume Delta)** | Binance recent trades | Whether buyers or sellers are the aggressive party. Rising CVD with rising price = conviction. Divergence = reversal warning. |
| **Funding Rate** | Binance futures | Cost of holding leveraged positions. Extreme positive = market too long = contrarian short. Extreme negative = contrarian long. |
| **Open Interest** | Binance futures | Rising price + rising OI = real trend. Rising price + falling OI = short squeeze (weaker, fades). |
| **Liquidation Cascades** | Binance futures | Forced selling/buying. The start of a cascade is a momentum signal — more liquidations trigger more liquidations. |
| **Taker Buy Ratio** | Binance klines | Fraction of volume that was buyer-initiated. Persistently high = accumulation. |

---

## Architecture

```
crypto_pipeline/
│
├── 1_data_collector.py       # Pulls live data from Binance Futures API
├── 2_feature_engineering.py  # Builds ML features + binary labels
├── 3_train_model.py          # Trains + evaluates multiple models
├── 4_live_predict.py         # Live directional prediction (BTC up/down)
├── 5_polymarket_arb.py       # Scans Polymarket for mispriced markets
│
├── data/                     # Collected snapshots, train/test CSVs
└── models/                   # Saved model, plots, backtest results
```

### What Each Script Does

**`1_data_collector.py`** — Hits Binance Futures public API (no API key needed) every 5 minutes and saves a snapshot of order book, recent trades for CVD, funding rate, open interest, liquidations, and OHLCV klines. Each snapshot is one row of training data.

**`2_feature_engineering.py`** — Converts raw snapshots into ML features. Computes deltas, ratios, rolling stats, confluence signals. Then labels each row: did price go UP or DOWN in the next 3 snapshots (15 minutes)? Uses time-based train/test split — never random split, which would leak future data into training.

**`3_train_model.py`** — Trains four models: Logistic Regression (baseline), Random Forest, Gradient Boosting, XGBoost. Evaluates on accuracy, ROC-AUC, log loss, and calibration. Runs a Kelly criterion backtest. Saves the best model.

**`4_live_predict.py`** — Loads the saved model, collects a fresh snapshot, and outputs a directional prediction with confidence. Used as input to the arbitrage detector.

**`5_polymarket_arb.py`** — The actual betting layer. Scans active Polymarket crypto markets, parses each question to extract price target and direction, computes fair probability using a log-normal model, compares against Polymarket's current yes/no price, combines with ML model signal, and flags opportunities where the edge exceeds 4%.

---

## Setup

### Requirements

```bash
pip install requests pandas numpy scikit-learn xgboost matplotlib scipy
```

No API keys required. All data sources are public.

```bash
# Create working directories
mkdir data models
```

---

## How to Use

### Phase 1 — Collect Data (weeks 1-3)

You need real data before you can train anything meaningful. Minimum 2 weeks, ideally 4-6.

```bash
# Test that the API connection works — single snapshot
python 1_data_collector.py

# Start continuous collection (every 5 minutes)
python 1_data_collector.py loop
```

Run this on a server or VPS so it continues while your machine is off:

```bash
# Run in background on Linux — survives terminal close
nohup python 1_data_collector.py loop > data/collector.log 2>&1 &

# Check it is still running
tail -f data/collector.log
```

Each day of collection = ~288 snapshots. Two weeks = ~4,000 rows.

### Phase 2 — Build Features and Train

Once you have at least 1-2 weeks of data:

```bash
# Engineer features and create train/test split
python 2_feature_engineering.py

# Train all models and evaluate
python 3_train_model.py
```

This outputs:
- `models/best_model.pkl` — the saved winning model
- `models/feature_importance_*.png` — which signals actually matter
- `models/calibration.png` — are the probabilities trustworthy?
- `models/backtest.png` — simulated PnL curve
- `models/model_comparison.csv` — all model metrics side by side

### Phase 3 — Live Prediction

```bash
# Single prediction right now
python 4_live_predict.py

# Continuous, every 5 minutes
python 4_live_predict.py --loop
```

### Phase 4 — Scan Polymarket for Edge

```bash
# List all active BTC/crypto markets on Polymarket
python 5_polymarket_arb.py --list

# Single scan — find mispriced markets right now
python 5_polymarket_arb.py

# Continuous monitoring every 5 minutes
python 5_polymarket_arb.py --monitor

# With your actual bankroll for Kelly sizing
python 5_polymarket_arb.py --monitor --bankroll 500
```

---

## Understanding Results

### Model Metrics

| Metric | What It Means | Good Threshold |
|--------|--------------|----------------|
| Accuracy | Raw % correct | >52% (random baseline is 50%) |
| ROC-AUC | Discrimination ability | >0.55 |
| Log Loss | Probability calibration | Lower than random baseline of 0.693 |
| Brier Score | Probability accuracy | <0.25 |

**Red flags — your model is broken:**
- Accuracy >70% on test set → almost certainly lookahead bias
- Perfect backtest curve → something is leaking future data
- Great on train, terrible on test → overfitting

**Reality check:** A consistently 53% accurate model with proper Kelly sizing is genuinely profitable over thousands of bets. The edge doesn't need to be large — it needs to be real and consistent.

### Polymarket Edge Signal

The arb detector flags an opportunity when three things align:

1. Fair probability (from log-normal model) differs from Polymarket price by more than 4%
2. ML model directional signal confirms the direction of the edge
3. Market has sufficient liquidity (more than $500)

When all three align — that is the highest-confidence bet signal.

### Kelly Bet Sizing

The system uses fractional Kelly (25% of full Kelly) which is conservative by design. Kelly tells you the mathematically optimal fraction of your bankroll to bet given your edge. At 25% Kelly you will grow slower but survive bad streaks. Never bet more than 5% of bankroll on a single market regardless of what Kelly outputs.

---

## What to Watch Out For

**Lookahead bias** is the silent killer. It makes your backtest look perfect while the live model fails completely. The only protection is strict time-based train/test splitting — never shuffle before splitting.

**Overfitting** happens when your model memorizes the training data instead of learning generalizable patterns. Signs: high training accuracy, low test accuracy, model performance degrades over time in production.

**Market regime changes.** A model trained during a low-volatility period may fail in high volatility and vice versa. Retrain regularly — at minimum monthly.

**Thin liquidity on Polymarket.** Even if you find edge, you may not be able to fill your full bet at the price you want. Size conservatively.

---

## Honest Assessment

This is hard. You are competing in a market with smart participants who also look for edge. A few things in your favor: Polymarket is less efficient than traditional financial markets — it is newer, smaller, and attracts more casual participants. Short-term crypto markets are genuinely noisy and manipulation-driven in ways that are patterned. And you have no latency constraint on Polymarket the way you would on Binance spot — the bet windows are minutes to hours, not milliseconds.

Treat this as a learning project for the first 3 months. Paper trade — track predictions without real money — until you have at least 200 predictions with a stable win rate. Only then consider real capital.

---

## Next Steps After This Works

1. News and sentiment signal — crypto Twitter moves markets minutes before price does
2. On-chain data — whale wallet movements are fully public on Bitcoin and Ethereum
3. Automated retraining pipeline — retrain weekly as new data accumulates
4. Multi-asset — extend beyond BTC to ETH, SOL markets on Polymarket
5. Sequential models — LSTM or Transformer that learns temporal patterns across snapshots rather than treating each snapshot independently