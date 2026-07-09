"""
api.py — FraudGuard AI Prediction Server
Hybrid ML Framework (Autoencoder + XGBoost) for Fraud Detection
Caleb University · CSC 400 · 22/10407
"""

from __future__ import annotations
import os, pickle, traceback
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_autoencoder = None
_xgb_model   = None
_metadata    = None
_explainer   = None

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')


def _build_autoencoder():
    """Rebuild exact same architecture as train.py — version-agnostic."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(30,), name='input')
    x   = tf.keras.layers.Dense(64, activation='relu', name='enc_1')(inp)
    x   = tf.keras.layers.BatchNormalization(name='batch_normalization')(x)
    x   = tf.keras.layers.Dense(32, activation='relu', name='enc_2')(x)
    x   = tf.keras.layers.Dense(16, activation='relu', name='enc_3')(x)
    bn  = tf.keras.layers.Dense(8,  activation='relu', name='bottleneck')(x)
    x   = tf.keras.layers.Dense(16, activation='relu', name='dec_1')(bn)
    x   = tf.keras.layers.Dense(32, activation='relu', name='dec_2')(x)
    x   = tf.keras.layers.Dense(64, activation='relu', name='dec_3')(x)
    out = tf.keras.layers.Dense(30, activation='linear', name='output')(x)
    return tf.keras.Model(inp, out, name='FraudGuard_Autoencoder')


def _load_models() -> bool:
    global _autoencoder, _xgb_model, _metadata, _explainer
    try:
        import tensorflow as tf
        import xgboost as xgblib
        import shap

        ae_path      = os.path.join(MODELS_DIR, 'autoencoder.keras')
        weights_path = os.path.join(MODELS_DIR, 'autoencoder.weights.h5')
        xgb_path     = os.path.join(MODELS_DIR, 'xgb_model.json')
        meta_path    = os.path.join(MODELS_DIR, 'metadata.pkl')

        for p in [xgb_path, meta_path]:
            if not os.path.exists(p):
                print(f"  [API] Missing: {p}")
                return False

        # ── Build architecture then load weights ──────────────────────────────
        # We rebuild in Python code (version-agnostic) then load only the
        # numerical weights — this avoids all Keras serialization issues.
        print("[API] Building Autoencoder architecture...")
        _autoencoder = _build_autoencoder()

        # Initialize weights by running one forward pass with dummy data
        _autoencoder(np.zeros((1, 30), dtype=np.float32))

        # Try weight sources in order of preference
        loaded = False

        # Source 1: load_weights from .keras file (skips architecture parsing)
        if os.path.exists(ae_path):
            try:
                _autoencoder.load_weights(ae_path)
                print("[API] Weights loaded from autoencoder.keras ✓")
                loaded = True
            except Exception as e1:
                print(f"  [API] .keras weights load failed: {e1}")

        # Source 2: separate .weights.h5 file
        if not loaded and os.path.exists(weights_path):
            try:
                _autoencoder.load_weights(weights_path)
                print("[API] Weights loaded from autoencoder.weights.h5 ✓")
                loaded = True
            except Exception as e2:
                print(f"  [API] .weights.h5 load failed: {e2}")

        # Source 3: extract weights from inside the .keras zip file
        if not loaded and os.path.exists(ae_path):
            try:
                import zipfile, tempfile
                print("[API] Extracting weights from .keras zip...")
                with tempfile.TemporaryDirectory() as tmp:
                    with zipfile.ZipFile(ae_path, 'r') as z:
                        z.extractall(tmp)
                    for candidate in ['model.weights.h5', 'weights.h5']:
                        wp = os.path.join(tmp, candidate)
                        if os.path.exists(wp):
                            _autoencoder.load_weights(wp)
                            print(f"[API] Weights extracted from zip: {candidate} ✓")
                            loaded = True
                            break
            except Exception as e3:
                print(f"  [API] zip extraction failed: {e3}")

        if not loaded:
            print("  [API] Could not load Autoencoder weights from any source.")
            return False

        # ── XGBoost ──────────────────────────────────────────────────────────
        print("[API] Loading XGBoost...")
        _xgb_model = xgblib.XGBClassifier()
        _xgb_model.load_model(xgb_path)

        # ── Metadata ─────────────────────────────────────────────────────────
        print("[API] Loading metadata...")
        with open(meta_path, 'rb') as f:
            _metadata = pickle.load(f)

        # ── SHAP ─────────────────────────────────────────────────────────────
        print("[API] Building SHAP explainer...")
        _explainer = shap.TreeExplainer(_xgb_model)

        print("[API] ✓ All models loaded successfully.")
        print(f"[API]   AE threshold : {_metadata['ae_threshold']:.6f}")
        return True

    except Exception as exc:
        print(f"  [API] Load error: {exc}")
        traceback.print_exc()
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield


app = FastAPI(
    title="FraudGuard AI — Prediction API",
    description="Hybrid ML Framework · Caleb University · CSC 400 · 22/10407",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)


class FormInput(BaseModel):
    amount:   float = Field(..., example=250000)
    type:     str   = Field(..., example="nip")
    hour:     int   = Field(..., example=2)
    auth:     str   = Field(..., example="otp")
    age:      int   = Field(..., example=6)
    freq:     int   = Field(..., example=8)
    device:   str   = Field(..., example="new")
    location: str   = Field(..., example="vpn")


class PredictionResult(BaseModel):
    fraud_probability:  float
    ae_score:           float
    xgb_score:          float
    risk_score:         int
    risk_level:         str
    shap_contributions: dict
    recommendation:     str
    mode:               str


def _models_ready():
    return _autoencoder is not None and _xgb_model is not None


def _risk_level(prob: float) -> str:
    if prob < 0.25: return 'LOW'
    if prob < 0.50: return 'MEDIUM'
    if prob < 0.75: return 'HIGH'
    return 'CRITICAL'


def _recommendation(level: str, amount: float, tx_type: str) -> str:
    t, a = tx_type.upper(), f"₦{amount:,.0f}"
    msgs = {
        'LOW':      f"✓ Transaction appears legitimate. Allow the {a} {t} to proceed.",
        'MEDIUM':   f"⚠ Elevated risk on {a} {t}. Request additional authentication before completing.",
        'HIGH':     f"⚡ High fraud risk. Hold {a} {t} pending manual review. Notify account holder.",
        'CRITICAL': f"🚨 CRITICAL — Block Immediately. Freeze {a} {t} and initiate account lockout. File STR with NIBSS."
    }
    return msgs.get(level, msgs['MEDIUM'])


def _form_to_features(data: FormInput) -> np.ndarray:
    fv = np.random.randn(30) * 0.25
    fv[28] = np.log1p(data.amount) / np.log1p(25_000) - 1.0
    fv[29] = (data.hour / 23.0) * 2 - 1.0
    auth_map = {'biometric': 0.0, 'pin_otp': 0.1, 'otp': 0.7, 'pin': 0.45, 'none': 2.8}
    a = auth_map.get(data.auth, 0.0)
    fv[13] -= a * 1.8; fv[9] -= a * 1.1
    if data.amount > 1_500_000:   fv[3] += 1.5; fv[10] -= 1.2
    elif data.amount > 500_000:   fv[3] += 0.9; fv[10] -= 0.7
    elif data.amount < 600:       fv[0] -= 1.8; fv[3]  -= 0.9
    if data.device == 'new':      fv[2] -= 1.1; fv[9]  -= 0.9
    if 0 <= data.hour <= 4:       fv[16] -= 1.4
    elif data.hour >= 22:         fv[16] -= 0.6
    loc_map = {'lagos':0,'abuja':0,'portharcourt':0,'other_ng':0.15,'international':0.9,'vpn':2.0}
    fv[11] -= loc_map.get(data.location, 0.0) * 1.0
    if data.freq >= 10:   fv[6] -= 1.6
    elif data.freq >= 6:  fv[6] -= 0.8
    elif data.freq >= 4:  fv[6] -= 0.4
    if data.age <= 1:     fv[20] -= 1.3
    elif data.age <= 6:   fv[20] -= 0.6
    return fv.reshape(1, -1)


def _run_prediction(features_30: np.ndarray):
    recon       = _autoencoder.predict(features_30, verbose=0)
    ae_mse      = float(np.mean((features_30 - recon) ** 2))
    ae_norm     = min(ae_mse / (_metadata['ae_threshold'] * 2), 1.0)
    features_31 = np.column_stack([features_30, [[ae_mse]]])
    xgb_proba   = float(_xgb_model.predict_proba(features_31)[0, 1])
    ensemble    = 0.35 * ae_norm + 0.65 * xgb_proba
    shap_vals   = _explainer.shap_values(features_31)[0]
    feat_names  = _metadata['feature_names']
    top6        = sorted(zip(feat_names, shap_vals.tolist()), key=lambda x: abs(x[1]), reverse=True)[:6]
    return ae_norm, xgb_proba, ensemble, {n: round(v, 4) for n, v in top6}


@app.get("/health")
async def health():
    loaded = _models_ready()
    return {
        "status":        "online" if loaded else "degraded",
        "models_loaded": loaded,
        "framework":     "Hybrid Autoencoder + XGBoost",
        "ae_threshold":  round(_metadata['ae_threshold'], 6) if loaded else None,
        "institution":   "Caleb University · CSC 400 · 22/10407"
    }


@app.post("/predict/demo", response_model=PredictionResult)
async def predict_demo(data: FormInput):
    if not _models_ready():
        raise HTTPException(503, "Models not loaded.")
    try:
        features_30                      = _form_to_features(data)
        ae_norm, xgb_p, ens, shap_dict  = _run_prediction(features_30)
        level                            = _risk_level(ens)
        return PredictionResult(
            fraud_probability  = round(ens, 4),
            ae_score           = round(ae_norm * 100, 1),
            xgb_score          = round(xgb_p  * 100, 1),
            risk_score         = int(round(ens * 100)),
            risk_level         = level,
            shap_contributions = shap_dict,
            recommendation     = _recommendation(level, data.amount, data.type),
            mode               = "real_model"
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/metrics")
async def get_metrics():
    if not _models_ready(): raise HTTPException(503, "Models not loaded.")
    return {"hybrid_ensemble": _metadata['hybrid_metrics'], "all_models": _metadata.get('all_results', {})}


@app.get("/shap")
async def get_shap_ranking():
    if not _models_ready(): raise HTTPException(503, "Models not loaded.")
    return {"shap_ranking": _metadata.get('shap_ranking', [])}
