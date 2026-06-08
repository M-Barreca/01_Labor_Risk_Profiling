"""
inference.py
------------
Standalone inference script. Loads saved model artefacts and scores a single record.

Usage:
    python src/inference.py                    # built-in sample
    python src/inference.py --input rec.json   # from JSON file
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ── helpers ───────────────────────────────────────────────────────────────────
EDUCATION_ORDER = {
    "Preschool":1,"1st-4th":2,"5th-6th":3,"7th-8th":4,"9th":5,"10th":6,
    "11th":7,"12th":8,"HS-grad":9,"Some-college":10,"Assoc-voc":11,
    "Assoc-acdm":12,"Bachelors":13,"Masters":14,"Prof-school":15,"Doctorate":16,
}

RARE_COUNTRIES = {
    "Holand-Netherlands","Scotland","Hungary","Honduras","Trinadad&Tobago",
    "Yugoslavia","Outlying-US(Guam-USVI-etc)","Cambodia",
}

def preprocess(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])
    if "native.country" in df.columns:
        df["native.country"] = df["native.country"].apply(
            lambda x: "Other" if x in RARE_COUNTRIES else x
        )
    return df


def predict(record: dict) -> dict:
    import joblib

    pipeline_path = MODELS_DIR / "pipeline.joblib"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Model not found at {pipeline_path}.\n"
            "Run the notebook and add a cell to save the pipeline:\n"
            "  import joblib\n"
            "  joblib.dump(grid_xgb.best_estimator_, 'models/pipeline.joblib')"
        )

    pipeline = joblib.load(pipeline_path)
    df       = preprocess(record)
    prob     = float(pipeline.predict_proba(df)[0][1])
    label    = pipeline.predict(df)[0]

    return {
        "income_above_50k": bool(label),
        "probability":      round(prob, 4),
        "risk_level":       "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
SAMPLE = {
    "age": 34, "workclass": "Private", "fnlwgt": 200000,
    "education": "Bachelors", "education.num": 13,
    "marital.status": "Married-civ-spouse", "occupation": "Prof-specialty",
    "relationship": "Husband", "race": "White", "sex": "Male",
    "capital.gain": 0, "capital.loss": 0,
    "hours.per.week": 45, "native.country": "United-States",
}

def main():
    parser = argparse.ArgumentParser(description="Labor Risk Profiling — Inference")
    parser.add_argument("--input", default=None, help="Path to JSON input file")
    args = parser.parse_args()

    record = json.load(open(args.input)) if args.input else SAMPLE
    if not args.input:
        print("ℹ️  No --input provided. Using built-in sample.\n")

    print("📥 Input:")
    print(json.dumps(record, indent=2))
    print()

    try:
        result = predict(record)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    icons = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    print("📤 Prediction:")
    print(f"  Income >50K:   {'Yes ✅' if result['income_above_50k'] else 'No ❌'}")
    print(f"  Probability:   {result['probability']:.1%}")
    print(f"  Risk level:    {icons[result['risk_level']]} {result['risk_level']}")

if __name__ == "__main__":
    main()
