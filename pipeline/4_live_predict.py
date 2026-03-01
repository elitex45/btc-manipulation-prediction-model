"""
LIVE PREDICTION
================
Loads trained model and makes live predictions on current market conditions.

Run AFTER 3_train_model.py

Usage:
    python 4_live_predict.py          # Single prediction
    python 4_live_predict.py --loop   # Continuous every 5 min
"""

import pickle
import pandas as pd
import numpy as np
import time
import argparse
from datetime import datetime

# Import our pipeline modules
import sys
sys.path.append(".")
from data_collector import collect_snapshot
from feature_engineering import engineer_features, get_feature_columns

MODELS_DIR = "./models"


def load_model():
    with open(f"{MODELS_DIR}/best_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_single(model_data):
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_cols"]
    model_name = model_data["model_name"]
    
    # Collect live data
    snapshot = collect_snapshot()
    df = pd.DataFrame([snapshot])
    
    # Engineer features
    df = engineer_features(df)
    
    # Get feature values
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    
    X = df[available_features].fillna(0)
    
    # Add missing features as 0
    for f in missing_features:
        X[f] = 0
    
    X = X[feature_cols]  # Ensure correct order
    
    # Scale if needed
    if model_name == "LogisticRegression":
        X_input = pd.DataFrame(scaler.transform(X), columns=feature_cols)
    else:
        X_input = X
    
    # Predict
    prob_up = model.predict_proba(X_input)[0][1]
    prob_down = 1 - prob_up
    prediction = "UP" if prob_up > 0.5 else "DOWN"
    confidence = max(prob_up, prob_down)
    
    # Signal strength
    if confidence > 0.65:
        strength = "STRONG"
    elif confidence > 0.55:
        strength = "MODERATE"
    else:
        strength = "WEAK (skip)"
    
    print(f"\n{'='*50}")
    print(f"PREDICTION — {datetime.utcnow().isoformat()}")
    print(f"{'='*50}")
    print(f"  Model:       {model_name}")
    print(f"  Direction:   {prediction}")
    print(f"  Prob UP:     {prob_up:.3f} ({prob_up*100:.1f}%)")
    print(f"  Prob DOWN:   {prob_down:.3f} ({prob_down*100:.1f}%)")
    print(f"  Confidence:  {confidence:.3f} → {strength}")
    
    if "ob_mid_price" in snapshot:
        print(f"  Price:       ${snapshot['ob_mid_price']:,.2f}")
    if "ob_imbalance_ratio" in snapshot:
        imb = snapshot["ob_imbalance_ratio"]
        print(f"  OB Imbalance: {imb:.3f} ({'bullish' if imb > 0.5 else 'bearish'})")
    if "fund_funding_rate" in snapshot:
        print(f"  Funding:     {snapshot['fund_funding_rate_pct']:.4f}%")
    
    # Polymarket recommendation
    print(f"\n  → Polymarket action: ", end="")
    if confidence < 0.55:
        print("SKIP — confidence too low")
    elif prediction == "UP":
        print(f"BUY 'YES' on UP market (confidence: {confidence*100:.1f}%)")
    else:
        print(f"BUY 'YES' on DOWN market (confidence: {confidence*100:.1f}%)")
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction,
        "prob_up": prob_up,
        "confidence": confidence,
        "strength": strength
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Run continuously every 5 minutes")
    args = parser.parse_args()
    
    print("Loading model...")
    model_data = load_model()
    print(f"Loaded: {model_data['model_name']}")
    print(f"Metrics: AUC={model_data['metrics']['roc_auc']:.4f}, Acc={model_data['metrics']['accuracy']:.4f}")
    
    if args.loop:
        print("\nRunning continuous predictions (Ctrl+C to stop)...")
        predictions = []
        while True:
            try:
                result = predict_single(model_data)
                predictions.append(result)
                pd.DataFrame(predictions).to_csv("./data/live_predictions.csv", index=False)
                time.sleep(300)  # 5 minutes
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        predict_single(model_data)