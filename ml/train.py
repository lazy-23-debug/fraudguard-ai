"""
train.py
═══════════════════════════════════════════════════════════════════
Hybrid ML Framework for Fraud Detection in Nigerian Digital
Payment Systems — Full Training Pipeline

Architecture  : Autoencoder (unsupervised) + XGBoost (supervised)
                stacking ensemble (Hybrid Framework)
Dataset       : ULB Credit Card Fraud Detection Dataset
                284,807 transactions · 492 fraud cases (0.17%)
                Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Methodology   : Design Science Research (DSR)
                Chapter 3 — Okeleke George Chukwudumebi
                Caleb University · CSC 400 · 22/10407

USAGE
─────
1. Place creditcard.csv in this folder  (ml/)
2. pip install -r requirements.txt
3. python train.py
4. Models saved to  ml/models/
5. Start API with:  uvicorn api:app --reload
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models    import Model
from tensorflow.keras.layers    import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# ══════════════════════════════════════════════════════════════════
#  BANNER
# ══════════════════════════════════════════════════════════════════

def banner(text, char='═', width=65):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

def step(n, text):
    print(f"\n[STEP {n}/9] {text}")
    print("─" * 50)

banner("HYBRID ML FRAMEWORK — TRAINING PIPELINE")
print("  Autoencoder + XGBoost Ensemble | ULB Dataset")
print("  Caleb University · CSC 400 · 22/10407")

# ══════════════════════════════════════════════════════════════════
#  STEP 1: LOAD DATASET
# ══════════════════════════════════════════════════════════════════
step(1, "Loading ULB Credit Card Dataset")

DATA_FILE = os.path.join(os.path.dirname(__file__), 'creditcard.csv')
if not os.path.exists(DATA_FILE):
    print(f"\n  ERROR: creditcard.csv not found at:\n  {DATA_FILE}")
    print("\n  Download it from:")
    print("  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("\n  Then place it in the ml/ folder and run this script again.")
    sys.exit(1)

df = pd.read_csv(DATA_FILE)

total      = len(df)
n_fraud    = df['Class'].sum()
n_legit    = total - n_fraud
fraud_pct  = (n_fraud / total) * 100

print(f"  Total transactions : {total:,}")
print(f"  Legitimate         : {n_legit:,}  ({100-fraud_pct:.2f}%)")
print(f"  Fraudulent         : {n_fraud:,}   ({fraud_pct:.4f}%)")
print(f"  Class imbalance    : {int(n_legit/n_fraud)}:1 (legitimate:fraud)")
print(f"  Amount range       : ₦{df['Amount'].min():.2f} – ₦{df['Amount'].max():.2f}")
print(f"  Time span          : {df['Time'].max()/3600:.1f} hours")

# ══════════════════════════════════════════════════════════════════
#  STEP 2: PREPROCESSING
# ══════════════════════════════════════════════════════════════════
step(2, "Feature Engineering & Preprocessing")

# Standardize Amount and Time
# (V1–V28 are already PCA-transformed in the ULB dataset)
amount_scaler = StandardScaler()
time_scaler   = StandardScaler()

df['Amount_scaled'] = amount_scaler.fit_transform(df[['Amount']])
df['Time_scaled']   = time_scaler.fit_transform(df[['Time']])

# Feature set: 30 features (V1-V28 + Amount_scaled + Time_scaled)
FEATURE_COLS = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']
X = df[FEATURE_COLS].values
y = df['Class'].values

print(f"  Feature matrix     : {X.shape}")
print(f"  Features used      : V1–V28 + Amount_scaled + Time_scaled")

# ── Train / Test split (stratified to preserve fraud ratio) ──────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\n  Train set: {X_train.shape[0]:,}  ({y_train.sum()} fraud)")
print(f"  Test set : {X_test.shape[0]:,}   ({y_test.sum()} fraud)")

# Save scalers for use by the API
os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

with open(os.path.join(MODELS_DIR, 'scalers.pkl'), 'wb') as f:
    pickle.dump({'amount': amount_scaler, 'time': time_scaler}, f)
print(f"\n  Scalers saved → models/scalers.pkl")

# ══════════════════════════════════════════════════════════════════
#  STEP 3: AUTOENCODER — Unsupervised Anomaly Detection
# ══════════════════════════════════════════════════════════════════
step(3, "Training Autoencoder (unsupervised anomaly detector)")

# The Autoencoder learns the normal distribution of LEGITIMATE
# transactions. High reconstruction error = anomalous = likely fraud.
X_train_legit = X_train[y_train == 0]
print(f"  Training on {X_train_legit.shape[0]:,} legitimate transactions only")

INPUT_DIM = X_train_legit.shape[1]  # 30

def build_autoencoder(input_dim: int) -> Model:
    """
    Stacked Autoencoder:
      Encoder: 30 → 64 → 32 → 16 → 8 (bottleneck)
      Decoder:  8 → 16 → 32 → 64 → 30
    """
    inputs = Input(shape=(input_dim,), name='input')

    # ── Encoder ──────────────────────────────────────────────────
    x = Dense(64, activation='relu', name='enc_1')(inputs)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu', name='enc_2')(x)
    x = Dense(16, activation='relu', name='enc_3')(x)
    bottleneck = Dense(8, activation='relu', name='bottleneck')(x)

    # ── Decoder ──────────────────────────────────────────────────
    x = Dense(16, activation='relu', name='dec_1')(bottleneck)
    x = Dense(32, activation='relu', name='dec_2')(x)
    x = Dense(64, activation='relu', name='dec_3')(x)
    outputs = Dense(input_dim, activation='linear', name='output')(x)

    return Model(inputs, outputs, name='FraudGuard_Autoencoder')


autoencoder = build_autoencoder(INPUT_DIM)
autoencoder.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse'
)
autoencoder.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-5, verbose=1)
]

t0 = time.time()
history = autoencoder.fit(
    X_train_legit, X_train_legit,
    epochs=100,
    batch_size=256,
    validation_split=0.10,
    callbacks=callbacks,
    verbose=1
)
print(f"\n  Training time : {time.time()-t0:.1f}s")
print(f"  Stopped at epoch {len(history.history['loss'])}")
print(f"  Final val_loss: {history.history['val_loss'][-1]:.6f}")

# ══════════════════════════════════════════════════════════════════
#  STEP 4: COMPUTE RECONSTRUCTION ERROR → Feature V31
# ══════════════════════════════════════════════════════════════════
step(4, "Computing reconstruction errors → Feature V31")

# Threshold = 95th percentile of legitimate reconstruction errors
legit_recon   = autoencoder.predict(X_train_legit, verbose=0)
legit_mse     = np.mean((X_train_legit - legit_recon) ** 2, axis=1)
ae_threshold  = np.percentile(legit_mse, 95)

print(f"  AE threshold (95th percentile): {ae_threshold:.6f}")

# Compute V31 for all splits
def ae_error(X):
    recon = autoencoder.predict(X, verbose=0)
    return np.mean((X - recon) ** 2, axis=1)

ae_train = ae_error(X_train)
ae_test  = ae_error(X_test)

# Append V31 to make 31-feature matrices
X_train_31 = np.column_stack([X_train, ae_train])
X_test_31  = np.column_stack([X_test,  ae_test])

# Quick AE-only evaluation
ae_preds_test = (ae_test > ae_threshold).astype(int)
print(f"\n  AE alone — Recall   : {recall_score(y_test, ae_preds_test)*100:.1f}%")
print(f"  AE alone — Precision: {precision_score(y_test, ae_preds_test, zero_division=0)*100:.1f}%")
print(f"  AE alone — AUC-ROC  : {roc_auc_score(y_test, ae_test):.4f}")

# Save autoencoder
AE_PATH = os.path.join(MODELS_DIR, 'autoencoder.keras')
autoencoder.save(AE_PATH)
print(f"\n  Autoencoder saved → {AE_PATH}")
# ADD THIS LINE immediately after autoencoder.save(AE_PATH)
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'autoencoder.weights.h5')
autoencoder.save_weights(WEIGHTS_PATH)
print(f"  Weights saved → {WEIGHTS_PATH}")

# ══════════════════════════════════════════════════════════════════
#  STEP 5: SMOTE — Class Imbalance Handling
# ══════════════════════════════════════════════════════════════════
step(5, "Applying SMOTE to training data")

# Apply SMOTE only to training set (never touch test set)
smote = SMOTE(
    sampling_strategy=0.15,  # create fraud cases up to 15% of majority
    random_state=42,
    k_neighbors=5
)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_31, y_train)

print(f"  Before SMOTE: {y_train.sum()} fraud / {(y_train==0).sum():,} legit")
print(f"  After SMOTE : {y_train_balanced.sum()} fraud / {(y_train_balanced==0).sum():,} legit")
print(f"  New ratio   : 1:{int((y_train_balanced==0).sum()/y_train_balanced.sum())}")

# ══════════════════════════════════════════════════════════════════
#  STEP 6: BASELINE MODELS (for comparison table in Chapter 3)
# ══════════════════════════════════════════════════════════════════
step(6, "Training baseline models for comparison")

results = {}

def evaluate(name, y_true, y_pred, y_proba):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0
    results[name] = dict(
        accuracy=acc, precision=prec, recall=rec,
        f1=f1, auc_roc=auc, fpr=fpr
    )
    print(f"  {name:<28} Acc={acc*100:.1f}%  F1={f1*100:.1f}%  AUC={auc:.4f}  FPR={fpr*100:.1f}%")
    return results[name]

# ── Logistic Regression ───────────────────────────────────────────
print("  Training Logistic Regression...")
lr = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
lr.fit(X_train_balanced, y_train_balanced)
lr_proba = lr.predict_proba(X_test_31)[:, 1]
lr_pred  = lr.predict(X_test_31)
evaluate('Logistic Regression', y_test, lr_pred, lr_proba)

# ── Random Forest ─────────────────────────────────────────────────
print("  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train_balanced, y_train_balanced)
rf_proba = rf.predict_proba(X_test_31)[:, 1]
rf_pred  = rf.predict(X_test_31)
evaluate('Random Forest', y_test, rf_pred, rf_proba)

# ══════════════════════════════════════════════════════════════════
#  STEP 7: XGBOOST — Supervised Classifier (Primary Model)
# ══════════════════════════════════════════════════════════════════
step(7, "Training XGBoost classifier (primary supervised model)")

# Internal validation split for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_balanced, y_train_balanced,
    test_size=0.15, random_state=42, stratify=y_train_balanced
)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='aucpr',
    early_stopping_rounds=30,
    n_jobs=-1
)

t0 = time.time()
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"  Best iteration : {xgb_model.best_iteration}")
print(f"  Training time  : {time.time()-t0:.1f}s")

xgb_proba = xgb_model.predict_proba(X_test_31)[:, 1]
xgb_pred  = xgb_model.predict(X_test_31)
evaluate('XGBoost', y_test, xgb_pred, xgb_proba)

# ══════════════════════════════════════════════════════════════════
#  STEP 8: HYBRID ENSEMBLE EVALUATION
# ══════════════════════════════════════════════════════════════════
step(8, "Evaluating Hybrid Ensemble (AE 35% + XGBoost 65%)")

# Ensemble score = weighted combination
ae_norm       = np.clip(ae_test / (ae_threshold * 2), 0, 1)
hybrid_proba  = 0.35 * ae_norm + 0.65 * xgb_proba
hybrid_pred   = (hybrid_proba >= 0.5).astype(int)

hybrid_metrics = evaluate('Hybrid Ensemble', y_test, hybrid_pred, hybrid_proba)

# Full confusion matrix
cm = confusion_matrix(y_test, hybrid_pred)
tn, fp, fn, tp = cm.ravel()

banner("FINAL RESULTS — HYBRID ENSEMBLE", char='═')
print(f"\n  Accuracy    : {hybrid_metrics['accuracy']*100:.2f}%")
print(f"  Precision   : {hybrid_metrics['precision']*100:.2f}%")
print(f"  Recall      : {hybrid_metrics['recall']*100:.2f}%")
print(f"  F1-Score    : {hybrid_metrics['f1']*100:.2f}%")
print(f"  AUC-ROC     : {hybrid_metrics['auc_roc']:.4f}")
print(f"  FPR         : {hybrid_metrics['fpr']*100:.2f}%")
print(f"\n  Confusion Matrix:")
print(f"  ┌─────────────┬──────────┬──────────┐")
print(f"  │             │ Pred:Leg │ Pred:Frau│")
print(f"  ├─────────────┼──────────┼──────────┤")
print(f"  │ Actual: Leg │  TN={tn:6,}│  FP={fp:5,}│")
print(f"  │ Actual: Fra │  FN={fn:6} │  TP={tp:5} │")
print(f"  └─────────────┴──────────┴──────────┘")
print(f"\n{classification_report(y_test, hybrid_pred, target_names=['Legitimate','Fraud'])}")

# ══════════════════════════════════════════════════════════════════
#  STEP 9: SHAP EXPLAINABILITY + SAVE
# ══════════════════════════════════════════════════════════════════
step(9, "SHAP Feature Importance + Saving All Models")

FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + [
    'Amount_scaled', 'Time_scaled', 'V31_AE_Error'
]

print("  Computing SHAP values (this may take a moment)...")
explainer   = shap.TreeExplainer(xgb_model)
X_sample    = X_test_31[:1000]   # 1000 samples for speed
shap_values = explainer.shap_values(X_sample)

mean_shap = np.abs(shap_values).mean(axis=0)
shap_df   = pd.DataFrame({
    'Feature':   FEATURE_NAMES,
    'Mean_SHAP': mean_shap
}).sort_values('Mean_SHAP', ascending=False)

print("\n  Top 10 Features by Mean |SHAP| value:")
print("  " + "─" * 40)
for _, row in shap_df.head(10).iterrows():
    bar = '█' * int(row['Mean_SHAP'] / shap_df['Mean_SHAP'].max() * 25)
    print(f"  {row['Feature']:<18}  {row['Mean_SHAP']:.4f}  {bar}")

# Save SHAP bar chart
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(shap_df.head(12)['Feature'][::-1],
        shap_df.head(12)['Mean_SHAP'][::-1],
        color='#00D9FF', alpha=0.85)
ax.set_xlabel('Mean |SHAP| value', fontsize=11)
ax.set_title('Feature Importance — Hybrid Ensemble (SHAP)', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, 'shap_importance.png'), dpi=150)
plt.close()
print(f"  SHAP chart saved → models/shap_importance.png")

# ── Save XGBoost ──────────────────────────────────────────────────
XGB_PATH = os.path.join(MODELS_DIR, 'xgb_model.json')
xgb_model.save_model(XGB_PATH)
print(f"  XGBoost saved   → {XGB_PATH}")

# ── Save all metadata (threshold, metrics, feature names) ─────────
metadata = {
    'ae_threshold':   ae_threshold,
    'feature_names':  FEATURE_NAMES,
    'feature_cols':   FEATURE_COLS,
    'all_results':    results,
    'hybrid_metrics': hybrid_metrics,
    'shap_ranking':   shap_df.to_dict('records'),
    'training_date':  pd.Timestamp.now().isoformat()
}
META_PATH = os.path.join(MODELS_DIR, 'metadata.pkl')
with open(META_PATH, 'wb') as f:
    pickle.dump(metadata, f)
print(f"  Metadata saved  → {META_PATH}")

banner("TRAINING COMPLETE", char='═')
print(f"\n  Models directory : ml/models/")
print(f"  Files created    :")
print(f"    autoencoder.keras")
print(f"    xgb_model.json")
print(f"    scalers.pkl")
print(f"    metadata.pkl")
print(f"    shap_importance.png")
print(f"\n  Start the prediction API:")
print(f"    cd ml/")
print(f"    uvicorn api:app --reload --port 8000")
print(f"\n  Then open index.html — the web app auto-connects!\n")
