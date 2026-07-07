# FraudGuard AI — ML Backend Setup

**Hybrid ML Framework for Fraud Detection in Nigerian Digital Payment Systems**  
Caleb University · CSC 400 · Okeleke George Chukwudumebi · 22/10407

---

## What This Does

`train.py` implements the complete methodology from Chapter 3:

| Stage | Detail |
|---|---|
| Dataset | ULB Credit Card Fraud Detection (284,807 transactions, 492 fraud) |
| Preprocessing | StandardScaler on Amount + Time; V1–V28 are PCA-transformed |
| Imbalance fix | SMOTE (sampling_strategy=0.15, applied only to training set) |
| Autoencoder | 30→64→32→16→8→16→32→64→30; trained on legitimate only; V31 = reconstruction MSE |
| XGBoost | 500 trees, depth=6, 31 features (V1–V28 + Amount + Time + V31) |
| Ensemble | 0.35 × AE + 0.65 × XGBoost |
| Explainability | SHAP TreeExplainer, mean \|SHAP\| ranking |
| API | FastAPI with CORS for web app integration |

---

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (first download creditcard.csv — see below)
python train.py

# 3. Start the prediction API
uvicorn api:app --reload --port 8000
```

Open `index.html` in your browser — the web app auto-connects to the API and uses the **real trained model** for predictions instead of the JavaScript simulation.

---

## Dataset Download

1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** (you need a free Kaggle account)
3. Extract and place **`creditcard.csv`** inside this `ml/` folder

The CSV is ~144 MB. It contains 30 anonymised features (V1–V28, Amount, Time) and a Class column (0 = legitimate, 1 = fraud).

---

## File Structure After Training

```
ml/
├── train.py             ← Training pipeline (run this first)
├── api.py               ← FastAPI prediction server
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── creditcard.csv       ← ULB dataset (you download this)
└── models/              ← Created automatically by train.py
    ├── autoencoder.keras    ← Trained Autoencoder
    ├── xgb_model.json       ← Trained XGBoost
    ├── scalers.pkl          ← StandardScaler for Amount + Time
    ├── metadata.pkl         ← Threshold, metrics, SHAP rankings
    └── shap_importance.png  ← SHAP bar chart (saved figure)
```

---

## API Endpoints

Once `uvicorn api:app --reload` is running:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model status + AE threshold |
| `/predict/demo` | POST | Predict from web app form inputs |
| `/predict/raw` | POST | Predict from real V1–V28 feature vector |
| `/metrics` | GET | Training evaluation metrics |
| `/shap` | GET | Global SHAP feature ranking |
| `/docs` | GET | Swagger UI (auto-generated) |

### Example — Demo Prediction

```bash
curl -X POST http://localhost:8000/predict/demo \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 750000,
    "type": "nip",
    "hour": 2,
    "auth": "none",
    "age": 1,
    "freq": 12,
    "device": "new",
    "location": "vpn"
  }'
```

Response:
```json
{
  "fraud_probability": 0.8924,
  "ae_score": 87.3,
  "xgb_score": 91.2,
  "risk_score": 89,
  "risk_level": "CRITICAL",
  "shap_contributions": {
    "V14": -0.4821,
    "V31_AE_Error": 0.3914,
    "V10": -0.2873,
    "Amount_scaled": 0.1952,
    "V4": 0.1634,
    "V7": -0.1201
  },
  "recommendation": "🚨 CRITICAL — Block Immediately...",
  "mode": "real_model"
}
```

---

## How the Web App Connects

The updated `js/predictor.js` sends a `POST /predict/demo` request when you click **Run Hybrid ML Analysis**:

- **API online** → green badge, real ML prediction, actual SHAP values from trained model
- **API offline** → yellow badge "Simulation Mode", JavaScript heuristic scoring (same as before)

The frontend auto-detects which mode it's in by pinging `/health` every 30 seconds.

---

## Training Time Estimates

| Hardware | Approximate Time |
|---|---|
| Modern laptop (CPU only) | 8–15 minutes |
| With GPU (CUDA) | 2–4 minutes |
| Google Colab (free GPU) | 3–5 minutes |

---

## Common Issues

**`creditcard.csv not found`**  
→ Download from Kaggle and place it in the `ml/` folder.

**`pip install` errors on TensorFlow**  
→ Try: `pip install tensorflow-cpu` (CPU-only, smaller download)

**Port 8000 already in use**  
→ Run on a different port: `uvicorn api:app --port 8001`  
→ Then update `API_URL` in `js/predictor.js` to `http://127.0.0.1:8001`

**SHAP import error**  
→ `pip install shap --upgrade`
