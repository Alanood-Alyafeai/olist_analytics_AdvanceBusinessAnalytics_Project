"""
score.py — Production Scoring Script
Olist Late Delivery Prediction Model
DSAI 4103 — Business Analytics Course Project

Usage (CLI):
    python score.py --input new_orders.csv --output predictions.csv

Usage (API / import):
    from score import LateDeliveryScorer
    scorer = LateDeliveryScorer()
    predictions = scorer.predict(orders_df)

Flask API usage:
    python score.py --serve --port 5000
    POST http://localhost:5000/predict  (JSON body with order features)
"""

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
MODEL_PATH    = "late_delivery_model.pkl"  # relative to model/ directory
FEATURES_PATH = "model_features.json"    # relative to model/ directory
DEFAULT_THRESHOLD = 0.30   # Business-tuned threshold (maximises F1 on test set)

RISK_THRESHOLDS = {
    "low":      (0.00, 0.10),
    "medium":   (0.10, 0.25),
    "high":     (0.25, 0.40),
    "critical": (0.40, 1.01),
}

RISK_ACTIONS = {
    "low":      "No action required — order is low risk.",
    "medium":   "Monitor order progress; check carrier status.",
    "high":     "Notify operations team; consider carrier upgrade.",
    "critical": "Proactively contact customer; offer compensation voucher.",
}


# ──────────────────────────────────────────────────────────────────────────────
# SCORER CLASS
# ──────────────────────────────────────────────────────────────────────────────
class LateDeliveryScorer:
    """
    Production-ready scoring class for the Olist Late Delivery Prediction model.

    Attributes
    ----------
    model     : trained sklearn-compatible model (GradientBoostingClassifier)
    features  : list of feature names the model expects
    threshold : decision threshold for binary classification

    Examples
    --------
    >>> scorer = LateDeliveryScorer()
    >>> df = pd.read_csv("new_orders.csv")
    >>> result = scorer.predict(df)
    >>> print(result[["late_probability", "risk_tier", "recommended_action"]])
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        features_path: str = FEATURES_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(features_path, "r") as f:
            self.features = json.load(f)
        self.threshold = threshold
        print(f"[LateDeliveryScorer] Model loaded. "
              f"Features: {len(self.features)}  Threshold: {threshold:.2f}")

    # ── Input validation ────────────────────────────────────────────────────
    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(
                f"Input is missing required features: {missing}\n"
                f"Expected: {self.features}"
            )
        X = df[self.features].copy()
        # Fill any remaining NaNs with column medians (safe for real-time scoring)
        X = X.fillna(X.median(numeric_only=True))
        # Ensure numeric dtypes
        X = X.astype(float)
        return X

    # ── Risk tier assignment ─────────────────────────────────────────────────
    @staticmethod
    def _risk_tier(prob: float) -> tuple:
        for tier, (lo, hi) in RISK_THRESHOLDS.items():
            if lo <= prob < hi:
                return tier, RISK_ACTIONS[tier]
        return "critical", RISK_ACTIONS["critical"]

    # ── Main prediction method ───────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score new orders.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain all features listed in self.features.

        Returns
        -------
        pd.DataFrame with columns:
            late_probability   (float)  — model score [0, 1]
            is_late_pred       (int)    — binary prediction {0=OnTime, 1=Late}
            risk_tier          (str)    — {low, medium, high, critical}
            recommended_action (str)    — human-readable operations instruction
        """
        X = self._validate(df)
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= self.threshold).astype(int)
        tiers_actions = [self._risk_tier(p) for p in probs]

        return pd.DataFrame({
            "late_probability":   np.round(probs, 4),
            "is_late_pred":       preds,
            "risk_tier":          [t[0] for t in tiers_actions],
            "recommended_action": [t[1] for t in tiers_actions],
        }, index=df.index)

    # ── Single-order scoring (for real-time API) ─────────────────────────────
    def predict_single(self, order: dict) -> dict:
        """
        Score a single order provided as a dictionary.

        Parameters
        ----------
        order : dict  — feature name → value

        Returns
        -------
        dict with late_probability, is_late_pred, risk_tier, recommended_action
        """
        df = pd.DataFrame([order])
        result = self.predict(df)
        return result.iloc[0].to_dict()

    # ── Model summary ────────────────────────────────────────────────────────
    def summary(self) -> None:
        print("=" * 55)
        print("  Olist Late Delivery Model — Summary")
        print("=" * 55)
        print(f"  Model type : {type(self.model).__name__}")
        print(f"  Features   : {len(self.features)}")
        print(f"  Threshold  : {self.threshold:.2f}")
        print()
        print("  Feature list:")
        for i, f in enumerate(self.features, 1):
            print(f"    {i:2d}. {f}")
        print()
        print("  Risk tier thresholds:")
        for tier, (lo, hi) in RISK_THRESHOLDS.items():
            print(f"    {tier:<10}: prob in [{lo:.2f}, {hi:.2f})")
        print("=" * 55)


# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL FLASK API
# ──────────────────────────────────────────────────────────────────────────────
def create_flask_app(scorer: LateDeliveryScorer):
    """
    Create a minimal Flask REST API for real-time scoring.
    Usage: python score.py --serve --port 5000
    Then: POST /predict with JSON body {"features": {...}}
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask not installed. Run: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": type(scorer.model).__name__,
                        "features": len(scorer.features)})

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        if not data or "features" not in data:
            return jsonify({"error": "Expected JSON with 'features' key"}), 400
        try:
            result = scorer.predict_single(data["features"])
            return jsonify(result)
        except ValueError as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:
            return jsonify({"error": f"Scoring failed: {e}"}), 500

    @app.route("/predict_batch", methods=["POST"])
    def predict_batch():
        data = request.get_json(force=True)
        if not data or "records" not in data:
            return jsonify({"error": "Expected JSON with 'records' key (list of feature dicts)"}), 400
        try:
            df = pd.DataFrame(data["records"])
            result = scorer.predict(df)
            return jsonify(result.to_dict(orient="records"))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Olist Late Delivery Prediction — Scoring Script"
    )
    parser.add_argument("--input",     type=str, help="Input CSV file path")
    parser.add_argument("--output",    type=str, help="Output CSV file path",
                        default="predictions.csv")
    parser.add_argument("--model",     type=str, default=MODEL_PATH,
                        help="Path to model pickle file")
    parser.add_argument("--features",  type=str, default=FEATURES_PATH,
                        help="Path to features JSON file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Classification threshold (default 0.30)")
    parser.add_argument("--serve",     action="store_true",
                        help="Start Flask REST API server")
    parser.add_argument("--port",      type=int, default=5000)
    parser.add_argument("--summary",   action="store_true",
                        help="Print model summary and exit")

    args = parser.parse_args()

    scorer = LateDeliveryScorer(
        model_path=args.model,
        features_path=args.features,
        threshold=args.threshold,
    )

    if args.summary:
        scorer.summary()
        return

    if args.serve:
        app = create_flask_app(scorer)
        print(f"Starting API server at http://localhost:{args.port}")
        print(f"  GET  /health")
        print(f"  POST /predict        (single order JSON)")
        print(f"  POST /predict_batch  (list of orders JSON)")
        app.run(host="0.0.0.0", port=args.port, debug=False)
        return

    if args.input:
        print(f"Reading input: {args.input}")
        df_input = pd.read_csv(args.input)
        print(f"  {len(df_input):,} orders to score")
        predictions = scorer.predict(df_input)
        combined = pd.concat([df_input.reset_index(drop=True),
                              predictions.reset_index(drop=True)], axis=1)
        combined.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        print(predictions["risk_tier"].value_counts().to_string())
    else:
        # Demo mode: score the test set
        print("No input file provided. Running demo on model test sample...")
        scorer.summary()
        demo_data = {feat: 0.0 for feat in scorer.features}
        demo_data["estimated_delivery_days"] = 15
        demo_data["approval_delay_hours"] = 2
        demo_data["same_state_delivery"] = 0
        demo_data["seller_historical_late_rate"] = 0.35
        result = scorer.predict_single(demo_data)
        print("\nDemo prediction:")
        for k, v in result.items():
            print(f"  {k:<22}: {v}")


# NOTE: Run this script from the model/ directory:
#   cd model && python score.py --summary
# Or from the project root:
#   python model/score.py --model model/late_delivery_model.pkl --features model/model_features.json --summary

if __name__ == "__main__":
    main()
