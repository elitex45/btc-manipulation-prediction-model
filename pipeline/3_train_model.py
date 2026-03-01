"""
MODEL TRAINING + EVALUATION
=============================
Trains multiple models and evaluates them properly.

Run AFTER 2_feature_engineering.py

Usage:
    python 3_train_model.py
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    log_loss, brier_score_loss, roc_auc_score,
    confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

DATA_DIR = "./data"
MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
def load_prepared_data():
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    
    feature_cols = [c for c in train.columns if c.startswith("f_")]
    
    X_train = train[feature_cols].fillna(0)
    y_train = train["label"]
    X_test = test[feature_cols].fillna(0)
    y_test = test["label"]
    
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Features: {len(feature_cols)}")
    
    return X_train, y_train, X_test, y_test, feature_cols


# ─────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────
def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
    }
    
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )
    
    return models


# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    """
    For prediction markets, we care about:
    1. Accuracy — raw directional correctness
    2. Log loss — how well-calibrated are probabilities
    3. Brier score — probability calibration
    4. ROC-AUC — discrimination ability
    
    A model that says "70% chance UP" when it's really 50% is dangerous.
    Calibration matters as much as accuracy.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    
    print(f"\n{'='*50}")
    print(f"MODEL: {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f} (lower = better, random = {log_loss(y_test, np.full_like(y_prob, 0.5)):.4f})")
    print(f"  Brier Score: {metrics['brier_score']:.4f} (lower = better, perfect = 0)")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted DOWN  Predicted UP")
    print(f"  Actual DOWN   {cm[0,0]:>14}  {cm[0,1]:>12}")
    print(f"  Actual UP     {cm[1,0]:>14}  {cm[1,1]:>12}")
    
    # Edge calculation (for betting)
    # If model says UP with >60% confidence, how often is it right?
    high_conf_mask = y_prob > 0.60
    if high_conf_mask.sum() > 10:
        hc_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
        print(f"\n  High confidence (>60%) accuracy: {hc_acc:.4f} on {high_conf_mask.sum()} samples")
    
    return metrics, y_prob


# ─────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────
def plot_feature_importance(model, feature_cols, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    
    importance = pd.Series(
        model.feature_importances_, 
        index=feature_cols
    ).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importance.head(20).plot(kind="barh")
    plt.title(f"Feature Importance — {model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/feature_importance_{model_name}.png", dpi=100)
    plt.close()
    
    print(f"\n  Top 10 features:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:40s} {imp:.4f}")
    
    return importance


# ─────────────────────────────────────────
# CALIBRATION PLOT
# ─────────────────────────────────────────
def plot_calibration(all_probs, y_test, model_names):
    """
    Shows if model's probabilities are trustworthy.
    Perfect calibration = diagonal line.
    """
    plt.figure(figsize=(8, 6))
    
    for model_name, y_prob in all_probs.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_of_positives, 
                marker="o", label=model_name)
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/calibration.png", dpi=100)
    plt.close()
    print(f"\nCalibration plot saved to {MODELS_DIR}/calibration.png")


# ─────────────────────────────────────────
# BACKTESTING SIMULATION
# ─────────────────────────────────────────
def backtest_strategy(y_test, y_prob, threshold=0.55, initial_bankroll=1000):
    """
    Kelly criterion backtest for binary prediction markets (Polymarket).

    Rules:
    - Only bet when model confidence > threshold (i.e. prob > threshold OR prob < 1-threshold)
    - Bet direction = UP if prob > 0.5, DOWN if prob < 0.5
    - Bet size = fractional Kelly (25% of full Kelly) capped at 5% of bankroll
    - Win = direction was correct
    - Payoff = 1:1 (simplified, Polymarket binary markets near 50/50)

    Kelly for 1:1 odds: f = 2p - 1
    where p = probability of winning the bet (not raw model output)
    """
    results = []
    bankroll = initial_bankroll

    y_test_arr = np.array(y_test)
    y_prob_arr = np.array(y_prob)

    for i, (true_label, prob) in enumerate(zip(y_test_arr, y_prob_arr)):

        # Direction: bet UP if model says >50%, bet DOWN if <50%
        bet_on_up = bool(prob > 0.5)
        confidence = float(prob if bet_on_up else 1.0 - prob)  # Always >= 0.5

        # Skip low-confidence signals
        if confidence < threshold:
            results.append({
                "step": i, "bankroll": bankroll,
                "action": "skip", "confidence": confidence,
                "bet_size": 0, "won": None, "pnl": 0
            })
            continue

        # Full Kelly for 1:1 odds = 2p - 1
        full_kelly = 2.0 * confidence - 1.0
        # Use quarter Kelly to be conservative, cap at 5% of bankroll
        kelly_fraction = min(full_kelly * 0.25, 0.05)
        kelly_fraction = max(kelly_fraction, 0.005)  # Floor at 0.5%

        bet_size = bankroll * kelly_fraction

        # Outcome: did our direction match what actually happened?
        price_went_up = bool(int(true_label) == 1)
        won = bool(bet_on_up == price_went_up)

        pnl = float(bet_size if won else -bet_size)
        bankroll = max(bankroll + pnl, 0.0)

        results.append({
            "step": i,
            "bankroll": bankroll,
            "action": "up" if bet_on_up else "down",
            "confidence": round(confidence, 4),
            "bet_size": round(bet_size, 2),
            "won": won,
            "pnl": round(pnl, 2)
        })
    
    results_df = pd.DataFrame(results)
    bets = results_df[results_df["action"] != "skip"].copy()

    if len(bets) == 0:
        print("No bets placed - lower confidence threshold")
        return results_df

    win_rate = bets["won"].astype(bool).mean()
    total_pnl = bankroll - initial_bankroll
    roi = total_pnl / initial_bankroll * 100

    # Max drawdown
    peak = results_df["bankroll"].cummax()
    drawdown = (results_df["bankroll"] - peak) / (peak + 1e-10) * 100
    max_drawdown = drawdown.min()

    print(f"\n  === BACKTEST RESULTS (threshold={threshold}) ===")
    print(f"  Bets placed:    {len(bets)}/{len(results_df)} ({len(bets)/len(results_df)*100:.1f}%)")
    print(f"  Win rate:       {win_rate:.3f} ({win_rate*100:.1f}%)")
    print(f"  Starting:       ${initial_bankroll:.2f}")
    print(f"  Final:          ${bankroll:.2f}")
    print(f"  Total PnL:      ${total_pnl:.2f}")
    print(f"  ROI:            {roi:.1f}%")
    print(f"  Max Drawdown:   {max_drawdown:.1f}%")
    
    # Plot bankroll over time
    plt.figure(figsize=(10, 4))
    plt.plot(results_df["bankroll"])
    plt.axhline(y=initial_bankroll, color="r", linestyle="--", label="Starting bankroll")
    plt.xlabel("Steps")
    plt.ylabel("Bankroll ($)")
    plt.title(f"Backtest: Bankroll Over Time (Win Rate: {win_rate*100:.1f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODELS_DIR}/backtest.png", dpi=100)
    plt.close()
    
    return results_df


# ─────────────────────────────────────────
# SAVE BEST MODEL
# ─────────────────────────────────────────
def save_model(model, scaler, feature_cols, metrics, model_name):
    import pickle
    
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_name": model_name
    }
    
    path = f"{MODELS_DIR}/best_model.pkl"
    with open(path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to {path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=== MODEL TRAINING ===\n")
    
    X_train, y_train, X_test, y_test, feature_cols = load_prepared_data()
    
    # Scale features (important for LogisticRegression, less so for trees)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=feature_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=feature_cols
    )
    
    models = get_models()
    all_metrics = {}
    all_probs = {}
    best_model = None
    best_auc = 0
    best_model_name = None
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Use scaled for LR, unscaled for trees
        if model_name == "LogisticRegression":
            model.fit(X_train_scaled, y_train)
            metrics, y_prob = evaluate_model(model, X_test_scaled, y_test, model_name)
        else:
            model.fit(X_train, y_train)
            metrics, y_prob = evaluate_model(model, X_test, y_test, model_name)
        
        all_metrics[model_name] = metrics
        all_probs[model_name] = y_prob
        
        # Feature importance
        plot_feature_importance(model, feature_cols, model_name)
        
        # Track best
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = model
            best_model_name = model_name
    
    # Comparison
    print(f"\n\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    comparison = pd.DataFrame(all_metrics).T
    comparison = comparison.sort_values("roc_auc", ascending=False)
    print(comparison.to_string())
    
    print(f"\n✓ Best model: {best_model_name} (AUC={best_auc:.4f})")
    
    # Calibration plot
    plot_calibration(all_probs, y_test, list(models.keys()))
    
    # Backtest with best model
    print(f"\n=== BACKTESTING {best_model_name} ===")
    best_probs = all_probs[best_model_name]
    backtest_strategy(y_test.values, best_probs, threshold=0.55)
    
    # Save best model
    use_scaled = best_model_name == "LogisticRegression"
    save_model(best_model, scaler, feature_cols, all_metrics[best_model_name], best_model_name)
    
    # Save comparison
    comparison.to_csv(f"{MODELS_DIR}/model_comparison.csv")
    print(f"\nAll outputs saved to {MODELS_DIR}/")