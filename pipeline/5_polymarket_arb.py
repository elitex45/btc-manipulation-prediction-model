"""
POLYMARKET ARBITRAGE / MISPRICING DETECTOR
===========================================
Compares what Polymarket thinks the probability is
vs what Binance price + our model says it should be.

The gap = your edge.

Polymarket API docs: https://docs.polymarket.com
CLOB API: https://clob.polymarket.com

Install:
    pip install requests pandas py-clob-client

Usage:
    python 5_polymarket_arb.py --list              # List active BTC markets
    python 5_polymarket_arb.py --monitor           # Monitor for mispricing live
    python 5_polymarket_arb.py --market <id>       # Watch specific market
"""

import requests
import pandas as pd
import numpy as np
import time
import argparse
import pickle
import sys
from datetime import datetime, timezone

sys.path.append(".")
from data_collector import collect_snapshot, fetch_klines
from feature_engineering import engineer_features

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"   # Market metadata
POLYMARKET_CLOB_API  = "https://clob.polymarket.com"        # Order book / prices
MODELS_DIR = "./models"

# Mispricing threshold to flag a bet opportunity
EDGE_THRESHOLD = 0.04       # Flag if model disagrees by >4%
MIN_LIQUIDITY  = 500        # Minimum $500 liquidity in market
MIN_CONFIDENCE = 0.57       # Minimum model confidence to act


# ─────────────────────────────────────────
# 1. FETCH ACTIVE BTC/CRYPTO MARKETS
# ─────────────────────────────────────────
def fetch_crypto_markets(keyword="bitcoin"):
    """
    Fetch active Polymarket markets related to crypto price movements.
    These are the 'will BTC be above X by date Y' type markets.
    """
    url = f"{POLYMARKET_GAMMA_API}/markets"
    params = {
        "active": True,
        "closed": False,
        "tag_slug": "crypto",          # Crypto category
        "limit": 100,
    }

    r = requests.get(url, params=params, timeout=10)
    markets = r.json()

    if isinstance(markets, dict) and "markets" in markets:
        markets = markets["markets"]

    # Filter for price movement markets
    price_keywords = ["above", "below", "higher", "lower", "reach", "exceed",
                      "bitcoin", "btc", "eth", "crypto", "up", "down"]

    relevant = []
    for m in markets:
        question = m.get("question", "").lower()
        if any(kw in question for kw in price_keywords):
            relevant.append({
                "id":           m.get("id"),
                "condition_id": m.get("conditionId"),
                "question":     m.get("question"),
                "end_date":     m.get("endDate"),
                "volume":       float(m.get("volume", 0) or 0),
                "liquidity":    float(m.get("liquidity", 0) or 0),
                "tokens":       m.get("tokens", []),
            })

    relevant.sort(key=lambda x: x["volume"], reverse=True)
    return relevant


# ─────────────────────────────────────────
# 2. GET CURRENT YES/NO PRICES FROM CLOB
# ─────────────────────────────────────────
def fetch_market_prices(condition_id):
    """
    Fetch current best bid/ask for YES and NO tokens.

    In Polymarket:
    - YES price = implied probability market thinks event happens
    - NO price  = 1 - YES price (approximately)

    Prices are in USDC, range 0-1 (e.g. 0.65 = 65% chance YES)
    """
    url = f"{POLYMARKET_CLOB_API}/book"
    params = {"token_id": condition_id}

    try:
        r = requests.get(url, params=params, timeout=10)
        book = r.json()

        bids = book.get("bids", [])
        asks = book.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None

        mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,        # This is the implied probability
            "spread": spread,
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_market_prices_gamma(market_id):
    """
    Alternative: fetch prices from Gamma API directly.
    Returns the last trade price as implied probability.
    """
    url = f"{POLYMARKET_GAMMA_API}/markets/{market_id}"
    try:
        r = requests.get(url, timeout=10)
        m = r.json()

        tokens = m.get("tokens", [])
        yes_token = next((t for t in tokens if t.get("outcome", "").upper() == "YES"), None)
        no_token  = next((t for t in tokens if t.get("outcome", "").upper() == "NO"), None)

        return {
            "yes_price": float(yes_token["price"]) if yes_token else None,
            "no_price":  float(no_token["price"])  if no_token  else None,
            "yes_token_id": yes_token.get("token_id") if yes_token else None,
            "no_token_id":  no_token.get("token_id")  if no_token  else None,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────
# 3. PARSE MARKET QUESTION → PRICE TARGET
# ─────────────────────────────────────────
def parse_market_question(question):
    """
    Extract the implied price target from market question text.

    Examples:
    "Will BTC be above $100,000 on March 15?" → target=100000, direction="above"
    "Will Bitcoin reach $80k by end of month?" → target=80000, direction="above"
    "Will BTC close below $90,000 this week?"  → target=90000, direction="below"
    """
    import re

    question_lower = question.lower()

    # Detect direction
    if any(w in question_lower for w in ["above", "higher", "exceed", "reach", "over"]):
        direction = "above"
    elif any(w in question_lower for w in ["below", "lower", "under", "drop"]):
        direction = "below"
    else:
        direction = "unknown"

    # Extract price (handles $100,000 / $100k / 100000 formats)
    price_pattern = r'\$?([\d,]+\.?\d*)\s*[kK]?'
    matches = re.findall(price_pattern, question)

    target_price = None
    for match in matches:
        clean = match.replace(",", "")
        try:
            val = float(clean)
            # Handle 'k' suffix
            if re.search(r'\$?' + match.replace(".", r"\.") + r'\s*[kK]', question):
                val *= 1000
            # Sanity check — BTC price range
            if 1000 < val < 10_000_000:
                target_price = val
                break
        except:
            continue

    return {"direction": direction, "target_price": target_price}


# ─────────────────────────────────────────
# 4. COMPUTE FAIR PROBABILITY FROM BINANCE
# ─────────────────────────────────────────
def compute_fair_probability(target_price, direction, current_price, 
                              time_to_expiry_hours, volatility_pct_per_hour=0.3):
    """
    Compute what the probability SHOULD be based on:
    - Current BTC price
    - Target price in the market question
    - Time until market resolves
    - Historical volatility

    Uses a simplified log-normal model (Black-Scholes intuition).
    This is your "fair value" to compare against Polymarket's price.

    volatility_pct_per_hour: BTC moves ~0.3% per hour on average (adjust based on regime)
    """
    from scipy import stats

    if target_price is None or current_price is None:
        return None

    # Log-normal price model
    # Expected log return over T hours
    T = time_to_expiry_hours
    sigma = volatility_pct_per_hour / 100  # Convert to decimal per hour

    log_return = np.log(target_price / current_price)
    std = sigma * np.sqrt(T)

    if std == 0:
        return 1.0 if (direction == "above" and current_price > target_price) else 0.0

    # Probability price ends above target
    # Assuming zero drift (conservative, markets price in drift)
    z_score = log_return / std

    if direction == "above":
        prob = 1 - stats.norm.cdf(z_score)
    else:
        prob = stats.norm.cdf(z_score)

    return round(float(prob), 4)


# ─────────────────────────────────────────
# 5. COMPUTE MODEL-BASED PROBABILITY ADJUSTMENT
# ─────────────────────────────────────────
def get_model_signal():
    """
    Run our trained ML model on current market conditions.
    Returns directional signal and confidence.
    """
    try:
        with open(f"{MODELS_DIR}/best_model.pkl", "rb") as f:
            model_data = pickle.load(f)

        model       = model_data["model"]
        scaler      = model_data["scaler"]
        feature_cols = model_data["feature_cols"]
        model_name  = model_data["model_name"]

        snapshot = collect_snapshot()
        df = pd.DataFrame([snapshot])
        df = engineer_features(df)

        X = df[[f for f in feature_cols if f in df.columns]].fillna(0)
        for f in feature_cols:
            if f not in X.columns:
                X[f] = 0
        X = X[feature_cols]

        if model_name == "LogisticRegression":
            X = pd.DataFrame(scaler.transform(X), columns=feature_cols)

        prob_up = model.predict_proba(X)[0][1]
        current_price = snapshot.get("ob_mid_price")

        return {
            "prob_up": prob_up,
            "prob_down": 1 - prob_up,
            "direction": "up" if prob_up > 0.5 else "down",
            "confidence": max(prob_up, 1 - prob_up),
            "current_price": current_price,
        }

    except FileNotFoundError:
        print("  [!] No trained model found. Run 3_train_model.py first.")
        print("      Using price-only signals for now.\n")
        return None


# ─────────────────────────────────────────
# 6. COMPUTE EDGE (MISPRICING)
# ─────────────────────────────────────────
def compute_edge(polymarket_prob, fair_prob, model_signal, direction):
    """
    Edge = difference between what Polymarket prices and what we think is fair.

    Three components:
    1. Statistical fair value (log-normal model)
    2. Model directional signal (our ML)
    3. Their combination

    Positive edge on YES = Polymarket underpricing YES → buy YES
    Negative edge on YES = Polymarket overpricing YES → buy NO
    """
    if polymarket_prob is None or fair_prob is None:
        return None

    # Raw mispricing
    raw_edge = fair_prob - polymarket_prob

    # Adjust with model signal if available
    model_adjustment = 0
    if model_signal and model_signal["confidence"] > MIN_CONFIDENCE:
        model_up = model_signal["prob_up"]
        # If market is "above" and model says up → increases fair prob
        if direction == "above":
            model_adjustment = (model_up - 0.5) * 0.1  # Small adjustment
        else:
            model_adjustment = (0.5 - model_up) * 0.1

    adjusted_edge = raw_edge + model_adjustment

    return {
        "raw_edge": round(raw_edge, 4),
        "model_adjustment": round(model_adjustment, 4),
        "total_edge": round(adjusted_edge, 4),
        "action": (
            "BUY YES" if adjusted_edge > EDGE_THRESHOLD else
            "BUY NO"  if adjusted_edge < -EDGE_THRESHOLD else
            "SKIP"
        ),
        "edge_pct": round(adjusted_edge * 100, 2),
    }


# ─────────────────────────────────────────
# 7. KELLY CRITERION BET SIZING
# ─────────────────────────────────────────
def kelly_bet_size(edge, win_prob, bankroll, max_fraction=0.05):
    """
    Kelly Criterion: optimal bet size given edge.

    For binary markets (Polymarket):
    f = (p*(b+1) - 1) / b
    where:
        p = win probability
        b = net odds (1:1 for binary → b=1)

    We use fractional Kelly (25% of full Kelly) to be conservative.
    """
    if abs(edge) < EDGE_THRESHOLD:
        return 0

    b = 1 / (1 - win_prob) - 1 if win_prob < 1 else 10  # Implied odds
    kelly = (win_prob * (b + 1) - 1) / b
    fractional_kelly = kelly * 0.25  # Quarter Kelly = conservative

    bet = bankroll * min(fractional_kelly, max_fraction)
    return max(0, round(bet, 2))


# ─────────────────────────────────────────
# 8. FULL MARKET ANALYSIS
# ─────────────────────────────────────────
def analyze_market(market, model_signal, current_price, time_now):
    """
    Full analysis pipeline for a single Polymarket market.
    """
    question = market["question"]
    market_id = market["id"]

    # Parse what the market is asking
    parsed = parse_market_question(question)
    target_price = parsed["target_price"]
    direction = parsed["direction"]

    # Get Polymarket's current implied probability
    prices = fetch_market_prices_gamma(market_id)
    poly_prob = prices.get("yes_price")

    # Time to expiry
    end_date = market.get("end_date")
    hours_to_expiry = None
    if end_date:
        try:
            end_dt = pd.to_datetime(end_date, utc=True)
            now_dt = pd.Timestamp.now(tz="UTC")
            hours_to_expiry = max(0, (end_dt - now_dt).total_seconds() / 3600)
        except:
            pass

    # Compute fair probability
    fair_prob = None
    if target_price and current_price and hours_to_expiry:
        fair_prob = compute_fair_probability(
            target_price, direction, current_price, hours_to_expiry
        )

    # Compute edge
    edge = compute_edge(poly_prob, fair_prob, model_signal, direction)

    result = {
        "market_id":        market_id,
        "question":         question[:80],
        "direction":        direction,
        "target_price":     target_price,
        "hours_to_expiry":  round(hours_to_expiry, 1) if hours_to_expiry else None,
        "poly_prob":        poly_prob,
        "fair_prob":        fair_prob,
        "liquidity":        market.get("liquidity"),
        "volume":           market.get("volume"),
        "edge":             edge,
    }

    return result


# ─────────────────────────────────────────
# 9. PRINT OPPORTUNITY
# ─────────────────────────────────────────
def print_opportunity(result, bankroll=1000):
    edge = result.get("edge")
    if not edge:
        return

    action = edge.get("action", "SKIP")
    edge_pct = edge.get("edge_pct", 0)

    # Color coding (terminal)
    if action == "BUY YES":
        tag = "🟢 BUY YES"
    elif action == "BUY NO":
        tag = "🔴 BUY NO "
    else:
        tag = "⬜ SKIP   "

    print(f"\n{'─'*70}")
    print(f"{tag} | Edge: {edge_pct:+.1f}%")
    print(f"  Question:    {result['question']}")
    print(f"  Target:      ${result['target_price']:,.0f} ({result['direction']})" if result['target_price'] else "  Target: (unparsed)")
    print(f"  Expires in:  {result['hours_to_expiry']:.1f}h" if result['hours_to_expiry'] else "  Expires: unknown")
    print(f"  Poly price:  {result['poly_prob']:.3f} ({result['poly_prob']*100:.1f}%)" if result['poly_prob'] else "  Poly price: N/A")
    print(f"  Fair value:  {result['fair_prob']:.3f} ({result['fair_prob']*100:.1f}%)" if result['fair_prob'] else "  Fair value: N/A")
    print(f"  Liquidity:   ${result['liquidity']:,.0f}" if result['liquidity'] else "")

    if action != "SKIP" and result['poly_prob']:
        win_prob = result['poly_prob'] if action == "BUY YES" else 1 - result['poly_prob']
        bet = kelly_bet_size(abs(edge['total_edge']), win_prob, bankroll)
        if bet > 0:
            print(f"  Kelly bet:   ${bet:.2f} (on ${bankroll} bankroll)")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list",    action="store_true", help="List active crypto markets")
    parser.add_argument("--monitor", action="store_true", help="Monitor for mispricing every 5 min")
    parser.add_argument("--market",  type=str, default=None, help="Analyze specific market ID")
    parser.add_argument("--bankroll",type=float, default=1000, help="Your bankroll for Kelly sizing")
    args = parser.parse_args()

    print("=== POLYMARKET MISPRICING DETECTOR ===\n")

    # Fetch markets
    print("Fetching active crypto markets from Polymarket...")
    markets = fetch_crypto_markets()
    print(f"Found {len(markets)} relevant markets\n")

    if args.list:
        print(f"{'ID':<10} {'Volume':>10} {'Liquidity':>10}  Question")
        print("─" * 80)
        for m in markets[:30]:
            print(f"{str(m['id']):<10} ${m['volume']:>9,.0f} ${m['liquidity']:>9,.0f}  {m['question'][:55]}")
        sys.exit(0)

    # Filter by liquidity
    liquid_markets = [m for m in markets if m.get("liquidity", 0) >= MIN_LIQUIDITY]
    print(f"Markets with >${MIN_LIQUIDITY} liquidity: {len(liquid_markets)}")

    if args.market:
        liquid_markets = [m for m in liquid_markets if str(m["id"]) == args.market]
        if not liquid_markets:
            print(f"Market {args.market} not found or below liquidity threshold.")
            sys.exit(1)

    def run_analysis():
        print(f"\n[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] Analyzing markets...\n")

        # Get model signal + current price (once, reuse for all markets)
        model_signal = get_model_signal()
        current_price = model_signal["current_price"] if model_signal else None

        if current_price:
            print(f"Current BTC: ${current_price:,.2f}")
        if model_signal:
            print(f"Model signal: {model_signal['direction'].upper()} "
                  f"(confidence: {model_signal['confidence']:.2%})")

        opportunities = []
        for market in liquid_markets[:20]:  # Analyze top 20 by volume
            try:
                result = analyze_market(market, model_signal, current_price, datetime.utcnow())
                opportunities.append(result)
                edge = result.get("edge")
                if edge and edge.get("action") != "SKIP":
                    print_opportunity(result, args.bankroll)
                time.sleep(0.3)  # Rate limit
            except Exception as e:
                print(f"  Error analyzing {market['id']}: {e}")

        # Summary
        actionable = [o for o in opportunities if o.get("edge") and o["edge"].get("action") != "SKIP"]
        print(f"\n{'─'*70}")
        print(f"Summary: {len(actionable)} opportunities found out of {len(opportunities)} markets analyzed")

        if not actionable:
            print("No clear edge detected. Markets appear fairly priced right now.")

        # Save results
        pd.DataFrame(opportunities).to_csv("./data/polymarket_analysis.csv", index=False)

    if args.monitor:
        print("Monitoring mode — running every 5 minutes (Ctrl+C to stop)\n")
        while True:
            try:
                run_analysis()
                time.sleep(300)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        run_analysis()