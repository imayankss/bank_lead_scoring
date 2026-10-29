"""Production propensity: the LightGBM chosen on validation, calibrated on validation."""

from __future__ import annotations

import json

import lightgbm as lgb
import pandas as pd

from lead_scoring.blend import blend, calibrate
from lead_scoring.paths import PROPENSITY_CHOICE_FILE, PROPENSITY_LGBM_FILE


def load_choice() -> dict:
    if not PROPENSITY_CHOICE_FILE.exists():
        raise FileNotFoundError(f"{PROPENSITY_CHOICE_FILE} is missing; run scripts/validate_propensity.py first")
    return json.loads(PROPENSITY_CHOICE_FILE.read_text(encoding="utf-8"))


def score_propensity(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM score, its validation-month rank scale and the calibrated propensity."""
    choice = load_choice()
    if choice["blend"]["weight"] != 0:
        # shortcut: validation gave the GRU weight 0, so GRU scoring isn't wired in; add it if a blend ever wins.
        raise NotImplementedError("the chosen blend uses the GRU, which production scoring doesn't load")
    out = pd.DataFrame({"lgbm_score": lgb.Booster(model_file=str(PROPENSITY_LGBM_FILE)).predict(X)})
    out["blend_score"] = blend(out["lgbm_score"], None, 0.0)
    out["predicted_propensity"] = calibrate(out["blend_score"], choice["calibrator"])
    return out
