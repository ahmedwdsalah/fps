#!/usr/bin/env python3
"""
Predict best algorithm using channel flag -> channel model routing.

Usage:
  python3 scripts/predict_f1_by_channel_flag.py \
    --flag Speed \
    --features-json /tmp/features.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "models" / "f1_9_channel_models"
RESULTS_ROOT = ROOT / "results" / "f1_9_channel_models"
MANIFEST = RESULTS_ROOT / "manifest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--flag", required=True, help="Channel flag, e.g. Speed, RPM, DRS")
    p.add_argument("--features-json", type=Path, required=True, help="Path to feature dict JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}. Train models first.")

    manifest = json.loads(MANIFEST.read_text())
    cols = manifest["feature_columns"]
    flag = args.flag
    ch = manifest["channels"].get(flag)
    if not ch or ch.get("status") != "trained":
        raise SystemExit(f"No trained model for flag '{flag}'.")

    feat = json.loads(args.features_json.read_text())
    x = np.array([[float(feat[c]) for c in cols]], dtype=np.float64)

    model = xgb.XGBClassifier()
    model.load_model(ch["model_path"])
    classes = json.loads(Path(ch["classes_path"]).read_text())

    proba = model.predict_proba(x)[0]
    i = int(np.argmax(proba))
    best = classes[i]

    out = {
        "flag": flag,
        "predicted_best_algorithm": best,
        "probabilities": {classes[j]: float(proba[j]) for j in range(len(classes))},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

