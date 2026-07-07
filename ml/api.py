"""
api.py
═══════════════════════════════════════════════════════════════════
FraudGuard AI — FastAPI Prediction Server
Hybrid ML Framework (Autoencoder + XGBoost) for Fraud Detection

START:  cd ml/
        uvicorn api:app --reload --port 8000

DOCS:   http://localhost:8000/docs   (Swagger UI auto-generated)
TEST:   http://localhost:8000/health

CORS is open to all origins so the frontend HTML file can call
the API directly from a browser (file:// or localhost).

Caleb University · CSC 400 · 22/10407
═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import pickle
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Globals (models loaded once at startup) ─────────────────────
_autoencoder = None
_xgb_model   = None
_metadata    = None
_explainer   = None   # SHAP TreeExplainer

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ══════════════════════════════════════════════════════════════════
#  MODEL LOADER (called once at startup via lifespan)
# ══════════════════════════════════════════════════════════════════

def _load_models() -> bool:
    global _autoencoder, _xgb_model, _metadata, _explainer
    try:
        import tensorflow as tf
        import xgboost as xgblib
        import shap

        ae_path   = os.path.join(MODELS_DIR, 'autoencoder.keras')
        xgb_path  = os.path.join(MODELS_DIR, 'xgb_model.json')
        meta_path = os.path.join(MODELS_DIR, 'metadata.pkl')

        for p in [ae_path, xgb_path, meta_path]:
            if not os.path.exists(p):
                print(f"  [API] Model file missing: {p}")
                print("  [API] Run  python train.py  first.")
                return False

        print("[API] Loading Autoencoder...")
        _autoencoder = tf.keras.models.load_model(ae_path)

        print("[API] Loading XGBoost...")
        _xgb_model = xgblib.XGBClassifier()
        _xgb_model.load_model(xgb_path)

        print("[API] Loading metadata...")
        with open(meta_path, 'rb') as f:
            _metadata = pickle.load(f)

        print("[API] Building SHAP explainer...")
        _explainer = shap.TreeExplainer(_xgb_model)

        print("[API] ✓ All models loaded successfully.")
        print(f"[API]   AE threshold : {_metadata['ae_threshold']:.6f}")
        return True

    except Exception as exc:
        print(f"[API] Model loading error: {exc}")
        traceback.print_exc()
        return False


# ══════════════════════════════════════════════════════════════════
#  APP FACTORY
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup; clean up on shutdown."""
    _load_models()
    yield
    # (nothing to clean up)


app = FastAPI(
    title="FraudGuard AI — Prediction API",
    description=(
        "Hybrid ML Framework for Fraud Detection in Nigerian Digital Payment Systems. "
        "Autoencoder (unsupervised anomaly detection) + XGBoost (supervised classification). "
        "Built for Caleb University · CSC 400 · Okeleke George Chukwudumebi · 22/10407."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS — open so the frontend HTML can call this API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

# ══════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════

class FormInput(BaseModel):
    """
    Simplified transaction attributes from the web app demo form.
    These are mapped to approximate ULB-style feature vectors
    before being fed to the real trained models.
    """
    amount:   float = Field(...,  example=250000,  description="Transaction amount in Naira")
    type:     str   = Field(...,  example="nip",   description="nip | ussd | pos | atm | card")
    hour:     int   = Field(...,  example=2,        description="Hour of day 0-23")
    auth:     str   = Field(...,  example="otp",   description="biometric | pin_otp | otp | pin | none")
    age:      int   = Field(...,  example=6,        description="Account age in months")
    freq:     int   = Field(...,  example=8,        description="Transactions in last hour")
    device:   str   = Field(...,  example="new",   description="known | new")
    location: str   = Field(...,  example="vpn",   description="lagos | abuja | portharcourt | other_ng | international | vpn")


class RawInput(BaseModel):
    """
    Actual ULB dataset format: V1–V28 + Amount_scaled + Time_scaled.
    Use this endpoint for batch testing against the real dataset.
    """
    features: list[float] = Field(
        ..., min_length=30, max_length=30,
        description="30 feature values: V1-V28, Amount_scaled, Time_scaled"
    )


class PredictionResult(BaseModel):
    fraud_probability:  float
    ae_score:           float    # 0-100 normalised AE reconstruction error
    xgb_score:          float    # 0-100 XGBoost fraud probability %
    risk_score:         int      # 0-100 ensemble score
    risk_level:         str      # LOW | MEDIUM | HIGH | CRITICAL
    shap_contributions: dict     # top feature → SHAP value
    recommendation:     str
    mode:               str      # "real_model" always here


# ══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _models_ready() -> bool:
    return _autoencoder is not None and _xgb_model is not None


def _risk_level(prob: float) -> str:
    if prob < 0.25: return 'LOW'
    if prob < 0.50: return 'MEDIUM'
    if prob < 0.75: return 'HIGH'
    return 'CRITICAL'


def _recommendation(level: str, amount: float, tx_type: str) -> str:
    t = tx_type.upper()
    a = f"₦{amount:,.0f}"
    msgs = {
        'LOW':      f"✓ Transaction appears legitimate. Autoencoder reconstruction error is within the normal range for legitimate {t} transactions. XGBoost assigns a low fraud probability. Allow the {a} transfer to proceed — no additional friction required.",
        'MEDIUM':   f"⚠ Elevated risk detected on this {a} {t} transaction. Request an additional out-of-band authentication challenge (callback or second OTP) before completing the transfer. Route to Tier-2 review queue for manual confirmation.",
        'HIGH':     f"⚡ High fraud risk on {a} {t} transaction. Temporarily hold transaction pending manual review. Notify the account holder via registered contact. Document anomaly flags for the NIBSS fraud reporting system and do not release without explicit out-of-band customer reconfirmation.",
        'CRITICAL': f"🚨 CRITICAL — Block Immediately. Multiple high-risk indicators detected simultaneously on {a} {t}. Freeze transaction, suspend customer session, and initiate account lockout. Escalate to fraud operations team. File Suspicious Transaction Report (STR) with the NIBSS per CBN FRM 2023 guidelines."
    }
    return msgs.get(level, msgs['MEDIUM'])


def _form_to_feature_vector(data: FormInput) -> np.ndarray:
    """
    Maps simplified form inputs to a 30-dimensional ULB-style feature vector.

    Strategy: start from a legitimate transaction profile (V1-V28 near 0)
    then inject statistical perturbations toward known fraud cluster positions
    based on published SHAP analyses of the ULB dataset.

    This allows the REAL trained Autoencoder and XGBoost to evaluate
    demo form inputs using their learned decision boundaries.
    """
    # Base profile: legitimate transaction (all PCA features ≈ 0)
    fv = np.random.randn(30) * 0.25   # small Gaussian noise
    fraud_signal = 0.0

    # ── Amount (index 28) ───────────────────────────────────────
    # In ULB, Amount_scaled ≈ log-normalized
    fv[28] = np.log1p(data.amount) / np.log1p(25_000) - 1.0

    # ── Time / hour (index 29) ──────────────────────────────────
    fv[29] = (data.hour / 23.0) * 2 - 1.0   # map 0-23h → -1 to +1

    # ── Authentication risk → perturb V14 (index 13), V10 (9) ──
    # V14 is the #1 SHAP feature; negative values → fraud cluster
    auth_map = {
        'biometric': 0.0,
        'pin_otp':   0.1,
        'otp':       0.7,   # SIM-swap vulnerable
        'pin':       0.45,
        'none':      2.8    # extreme anomaly
    }
    a_factor = auth_map.get(data.auth, 0.0)
    fv[13] -= a_factor * 1.8   # V14 strongly negative in fraud
    fv[9]  -= a_factor * 1.1   # V10
    fraud_signal += a_factor * 0.32

    # ── Amount extremes → perturb V4 (3), V11 (10) ─────────────
    if data.amount > 1_500_000:
        fv[3]  += 1.5;  fv[10] -= 1.2;  fraud_signal += 0.45
    elif data.amount > 500_000:
        fv[3]  += 0.9;  fv[10] -= 0.7;  fraud_signal += 0.25
    elif data.amount < 600:         # probe transaction
        fv[0]  -= 1.8;  fv[3]  -= 0.9;  fraud_signal += 0.28

    # ── New / unknown device → perturb V3 (2), V10 (9) ─────────
    if data.device == 'new':
        fv[2]  -= 1.1;  fv[9]  -= 0.9;  fraud_signal += 0.30

    # ── Night time → perturb V17 (16) ───────────────────────────
    if 0 <= data.hour <= 4:
        fv[16] -= 1.4;  fraud_signal += 0.28
    elif data.hour >= 22:
        fv[16] -= 0.6;  fraud_signal += 0.10

    # ── Location risk → perturb V12 (11) ────────────────────────
    loc_map = {
        'lagos': 0, 'abuja': 0, 'portharcourt': 0,
        'other_ng': 0.15, 'international': 0.9, 'vpn': 2.0
    }
    l_factor = loc_map.get(data.location, 0.0)
    fv[11] -= l_factor * 1.0;  fraud_signal += l_factor * 0.28

    # ── High frequency → perturb V7 (6) ─────────────────────────
    if data.freq >= 10:
        fv[6]  -= 1.6;  fraud_signal += 0.35
    elif data.freq >= 6:
        fv[6]  -= 0.8;  fraud_signal += 0.18
    elif data.freq >= 4:
        fv[6]  -= 0.4;  fraud_signal += 0.08

    # ── New account → perturb V21 (20) ──────────────────────────
    if data.age <= 1:
        fv[20] -= 1.3;  fraud_signal += 0.32
    elif data.age <= 6:
        fv[20] -= 0.6;  fraud_signal += 0.12

    # ── NIP transfer type → perturb V17 slightly ────────────────
    type_map = {'nip': 0.2, 'ussd': 0.15, 'card': 0.12, 'atm': 0.05, 'pos': 0}
    fv[16] -= type_map.get(data.type, 0) * 0.3

    return fv.reshape(1, -1)


def _run_prediction(features_30: np.ndarray):
    """Core inference: AE error → V31 → XGBoost → ensemble."""
    # AE reconstruction error
    recon       = _autoencoder.predict(features_30, verbose=0)
    ae_mse      = float(np.mean((features_30 - recon) ** 2))
    ae_norm     = min(ae_mse / (_metadata['ae_threshold'] * 2), 1.0)

    # Append V31
    features_31 = np.column_stack([features_30, [[ae_mse]]])

    # XGBoost fraud probability
    xgb_proba   = float(_xgb_model.predict_proba(features_31)[0, 1])

    # Ensemble
    ensemble    = 0.35 * ae_norm + 0.65 * xgb_proba

    # SHAP (top 6 features only)
    shap_vals   = _explainer.shap_values(features_31)[0]
    feat_names  = _metadata['feature_names']
    top6        = sorted(
        zip(feat_names, shap_vals.tolist()),
        key=lambda x: abs(x[1]), reverse=True
    )[:6]
    shap_dict   = {name: round(val, 4) for name, val in top6}

    return ae_norm, xgb_proba, ensemble, shap_dict


# ══════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Health check — also tells the frontend whether models are loaded."""
    loaded = _models_ready()
    return {
        "status":        "online"  if loaded else "degraded",
        "models_loaded": loaded,
        "framework":     "Hybrid Autoencoder + XGBoost",
        "ae_threshold":  round(_metadata['ae_threshold'], 6) if loaded else None,
        "institution":   "Caleb University · CSC 400 · 22/10407"
    }


@app.post("/predict/demo", response_model=PredictionResult)
async def predict_demo(data: FormInput):
    """
    Predict fraud risk from the web app form inputs.

    Maps simplified transaction attributes (amount, auth method,
    device, location, time, frequency, account age) to an
    approximate ULB-style feature vector, then runs the real
    Autoencoder and XGBoost pipeline for a genuine ML prediction.
    """
    if not _models_ready():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run  python train.py  first, then restart the API."
        )

    try:
        features_30              = _form_to_feature_vector(data)
        ae_norm, xgb_p, ens, shap_dict = _run_prediction(features_30)
        level                    = _risk_level(ens)

        return PredictionResult(
            fraud_probability  = round(ens,   4),
            ae_score           = round(ae_norm  * 100, 1),
            xgb_score          = round(xgb_p   * 100, 1),
            risk_score         = int(round(ens * 100)),
            risk_level         = level,
            shap_contributions = shap_dict,
            recommendation     = _recommendation(level, data.amount, data.type),
            mode               = "real_model"
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/raw")
async def predict_raw(body: RawInput):
    """
    Predict from actual ULB dataset features (V1–V28, Amount_scaled, Time_scaled).
    Use this for batch evaluation or feeding real transaction vectors.
    """
    if not _models_ready():
        raise HTTPException(503, "Models not loaded.")

    try:
        features_30              = np.array(body.features).reshape(1, -1)
        ae_norm, xgb_p, ens, _  = _run_prediction(features_30)
        level                    = _risk_level(ens)

        return {
            "fraud_probability": round(ens,  4),
            "ae_score":          round(ae_norm * 100, 1),
            "xgb_score":         round(xgb_p  * 100, 1),
            "risk_score":        int(round(ens * 100)),
            "risk_level":        level
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
async def get_metrics():
    """Return the evaluation metrics recorded during training."""
    if not _models_ready():
        raise HTTPException(503, "Models not loaded.")
    return {
        "hybrid_ensemble": _metadata['hybrid_metrics'],
        "all_models":      _metadata.get('all_results', {}),
        "training_date":   _metadata.get('training_date', 'unknown')
    }


@app.get("/shap")
async def get_shap_ranking():
    """Return global SHAP feature importance ranking from training."""
    if not _models_ready():
        raise HTTPException(503, "Models not loaded.")
    return {"shap_ranking": _metadata.get('shap_ranking', [])}
