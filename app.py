import os
import io
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# Safe XGBoost import + fallback
# -------------------------------
XGB_AVAILABLE = True
XGB_IMPORT_ERROR = None
try:
    from xgboost import XGBClassifier
except Exception as e:
    XGB_AVAILABLE = False
    XGB_IMPORT_ERROR = e
    from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Exoplanet Classifier Pro", page_icon="🪐", layout="wide")

st.title("🪐 Exoplanet Classifier — Data Explorer, Grid Search & Final Evaluation")

with st.expander("About", expanded=False):
    st.markdown(
        """
        **What this app does**
        - Load `exoplanets.csv` (or upload your CSV)
        - Choose one of **three dataset scopes** for training: **Short subset**, **Loaded subset**, **All rows**
        - **Tune hyperparameters** with Grid or Randomized search using the exact value lists you specify
        - Show **Best params**, **Best score (recall)** and **All scores**
        - Train a final model and display full **classification report** (precision/recall/F1/support) and confusion matrix
        - Optional: run a **final-model style** evaluation where a model trained on one scope is evaluated on **All rows** (X_all vs y_all)
        """
    )

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data
def load_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

@st.cache_data
def load_csv_no_cache(path: str) -> pd.DataFrame:
    # separate cache key for full file (avoids collisions with nrows)
    return pd.read_csv(path)

def coerce_labels_to_binary(series: pd.Series) -> tuple[pd.Series, dict]:
    """Map labels to {0,1}. If values are {1,2} -> {0,1}. Otherwise LabelEncoder to 0..K-1."""
    y = series.copy()
    mapping = {}
    uniq = sorted(pd.Series(y).dropna().unique().tolist())

    if set(uniq) == {1, 2}:
        mapping = {1: 0, 2: 1}
        y = y.map(mapping)
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        mapping = {orig: enc for orig, enc in zip(le.classes_, range(len(le.classes_)))}
    return y.astype(int), mapping

def split_xy(df: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    assert label_col in df.columns, f"'{label_col}' not in columns."
    numeric = df.select_dtypes(include=[np.number]).copy()
    y_raw = df[label_col]
    if label_col in numeric.columns:
        numeric = numeric.drop(columns=[label_col])
    y_enc, mapping = coerce_labels_to_binary(y_raw)
    return numeric, y_enc, mapping

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return float(neg / pos) if pos > 0 else 1.0

@st.cache_data
def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T
    # Reorder common rows if available
    cols = [c for c in ["precision","recall","f1-score","support"] if c in df_report.columns]
    return df_report[cols]

# parse comma or space separated lists (ints/floats)
def parse_values(text: str):
    text = (text or "").strip()
    if not text:
        return None
    # Allow Python list literal or comma/space separated values
    try:
        # Try safe literal eval first (e.g., "[1,2,3]", "(0.1, 0.2)")
        val = ast.literal_eval(text)
        if isinstance(val, (list, tuple)):
            return list(val)
        else:
            return [val]
    except Exception:
        # Fallback: split on commas or spaces
        parts = [p for chunk in text.split(',') for p in chunk.split()]
        vals = []
        for p in parts:
            try:
                if "." in p or "e" in p.lower():
                    vals.append(float(p))
                else:
                    vals.append(int(p))
            except ValueError:
                st.warning(f"Could not parse value '{p}', skipping.")
        return vals or None

# -------------------------------
# Data selection
# -------------------------------
st.sidebar.header("Data")
DEFAULT_CSV = "exoplanets.csv"

source = st.sidebar.radio("Dataset source", ["Use local exoplanets.csv", "Upload CSV"], index=0)
rows_to_read = st.sidebar.number_input("Rows to read (0 = all)", min_value=0, value=400, step=50)

if source == "Upload CSV":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is None:
        st.stop()
    df_loaded = pd.read_csv(up, nrows=None if rows_to_read == 0 else rows_to_read)
else:
    if not os.path.exists(DEFAULT_CSV):
        st.error(f"Missing `{DEFAULT_CSV}` in this directory.")
        st.stop()
    df_loaded = load_csv(DEFAULT_CSV, None if rows_to_read == 0 else rows_to_read)

# Also prepare ALL rows from local file if available (used for final_model-style eval)
if os.path.exists(DEFAULT_CSV):
    df_allfile = load_csv_no_cache(DEFAULT_CSV)
else:
    df_allfile = df_loaded.copy()

# Choose label column
label_guess = next((c for c in df_loaded.columns if c.upper()=="LABEL"), df_loaded.columns[0])
label_col = st.sidebar.selectbox("Label column", options=df_loaded.columns.tolist(), index=df_loaded.columns.get_loc(label_guess))

st.subheader("Dataset preview")
st.dataframe(df_loaded.head(), use_container_width=True)
st.caption(f"Loaded subset shape: {df_loaded.shape[0]} rows × {df_loaded.shape[1]} columns")

# Build X,y for three scopes
X_loaded, y_loaded, mapping_loaded = split_xy(df_loaded, label_col)
X_short = X_loaded.iloc[:74, :].copy()
y_short = y_loaded.iloc[:74].copy()

X_all, y_all, mapping_all = split_xy(df_allfile, label_col)

colA, colB, colC = st.columns(3)
with colA:
    st.write("**Label mapping (loaded)**:", mapping_loaded)
    st.write("Counts:", dict(zip(*np.unique(y_loaded, return_counts=True))))
with colB:
    st.write("**Label mapping (all-file)**:", mapping_all)
    st.write("Counts:", dict(zip(*np.unique(y_all, return_counts=True))))
with colC:
    st.write("Short subset size:", X_short.shape)

# Choose which (X,y) to train with
scope = st.radio(
    "Training scope (X, y):",
    ["Short subset (first 74)", "Loaded subset (current table)", "All rows (full file)"]
)
if scope.startswith("Short"):
    X_used, y_used = X_short, y_short
elif scope.startswith("Loaded"):
    X_used, y_used = X_loaded, y_loaded
else:
    X_used, y_used = X_all, y_all

# -------------------------------
# Model core + train/test split
# -------------------------------
st.sidebar.header("Train/Test split")
stratify = st.sidebar.checkbox("Stratify by label", value=True)
random_state = st.sidebar.number_input("Random state", 0, 10_000, 2, 1)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.25, 0.05)

if stratify and len(np.unique(y_used))>1:
    strat = y_used
else:
    strat = None

X_train, X_test, y_train, y_test = train_test_split(
    X_used, y_used, test_size=test_size, random_state=random_state, stratify=strat
)

spw_auto = compute_scale_pos_weight(y_train)

st.write("**Train/Test shapes** — ", f"X_train: {X_train.shape}, y_train: {y_train.shape}; X_test: {X_test.shape}, y_test: {y_test.shape}")
st.write(f"Auto `scale_pos_weight` (neg/pos on training): **{spw_auto:.3f}**")

# -------------------------------
# Base hyperparameters (for direct training)
# -------------------------------
st.sidebar.header("Base Model (for Train Now)")
use_xgb = st.sidebar.checkbox("Use XGBoost", value=True if XGB_AVAILABLE else False)
if use_xgb and not XGB_AVAILABLE:
    st.sidebar.error(f"XGBoost unavailable: {XGB_IMPORT_ERROR}")
    st.sidebar.info("Fallback model (HistGradientBoostingClassifier) will be used.")

n_estimators = st.sidebar.slider("n_estimators", 50, 1000, 300, 50)
max_depth    = st.sidebar.slider("max_depth", 1, 12, 3, 1)
learning_rate = st.sidebar.select_slider("learning_rate", options=[0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5,1.0], value=0.1)
subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
reg_lambda = st.sidebar.slider("reg_lambda (L2)", 0.0, 20.0, 1.0, 0.5)

spw_mode = st.sidebar.radio("Class weight", ["Auto (neg/pos)", "Manual"], horizontal=True)
manual_spw = st.sidebar.number_input("scale_pos_weight (binary)", min_value=0.0, value=float(spw_auto), step=0.5)

cv_splits = st.sidebar.slider("CV folds (for Grid/Random search)", 2, 5, 3, 1)

# -------------------------------
# Model builder
# -------------------------------
def make_model(**overrides):
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        scale_pos_weight=(spw_auto if spw_mode.startswith("Auto") else manual_spw),
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(overrides)

    if use_xgb and XGB_AVAILABLE:
        return XGBClassifier(**params)
    else:
        # Fallback approximations
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"] if params["max_depth"]>0 else None,
            max_iter=params["n_estimators"],
            l2_regularization=params["reg_lambda"],
            random_state=random_state,
        )

# -------------------------------
# Train now section
# -------------------------------
train_now = st.button("🚀 Train Now (base params)")
if train_now:
    model = make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:,1]
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Accuracy", f"{acc:.3f}")
        st.metric("Recall (pos=1)", f"{rec:.3f}")
    with m2:
        st.metric("Precision", f"{prec:.3f}")
        st.metric("F1", f"{f1:.3f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["0","1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["0","1"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    st.pyplot(fig)

    st.subheader("Classification report")
    st.dataframe(classification_report_df(y_test, y_pred), use_container_width=True)

# -------------------------------
# Light curve explorer
# -------------------------------
st.subheader("🔭 Light Curve Explorer")
row_idx = st.number_input("Row index (from training scope)", min_value=0, max_value=len(X_used)-1, value=1, step=1)
fig2, ax2 = plt.subplots(figsize=(9,3))
ax2.plot(np.arange(X_used.shape[1]), X_used.iloc[int(row_idx)])
ax2.set_xlabel("Number of Observations")
ax2.set_ylabel("Light Flux")
ax2.set_title(f"Light Plot — row {int(row_idx)}")
st.pyplot(fig2)

# -------------------------------
# Grid Search Playground
# -------------------------------
st.header("🎛️ Grid Search Playground")
st.caption("Select which hyperparameters to include. Unchecked ones stay at the base-model defaults. Choose Grid or Randomized search; then click **Run selected search**.")

# Seed session state for text inputs to avoid Streamlit warning
_defaults = {
    "gs_n_estimators": "50, 200, 400, 800",
    "gs_learning_rate": "0.4, 0.5, 0.6, 0.7, 1.0",
    "gs_max_depth": "1, 2, 4, 8",
    "gs_subsample": "0.3, 0.5, 0.7, 0.9",
    "gs_gamma": "0.05, 0.1, 0.5, 1",
    "gs_max_delta_step": "1, 3, 5, 7",
    "gs_colsample": "0.3, 0.5, 0.7, 0.9, 1",
}
for _k, _v in _defaults.items():
    st.session_state.setdefault(_k, _v)

# Which dataset to run grid search on
gs_scope = st.radio("Grid search uses:", ["Short subset", "Loaded subset", "All rows"], horizontal=True)
if gs_scope.startswith("Short"):
    X_gs, y_gs = X_short, y_short
elif gs_scope.startswith("Loaded"):
    X_gs, y_gs = X_loaded, y_loaded
else:
    X_gs, y_gs = X_all, y_all

left_cols, right_cols = st.columns(2)
with left_cols:
    inc_n_estimators = st.checkbox("Include n_estimators", value=True)
    st.text_input("n_estimators list", key="gs_n_estimators", disabled=not inc_n_estimators)

    inc_learning_rate = st.checkbox("Include learning_rate", value=True)
    st.text_input("learning_rate list", key="gs_learning_rate", disabled=not inc_learning_rate)

    inc_max_depth = st.checkbox("Include max_depth", value=True)
    st.text_input("max_depth list", key="gs_max_depth", disabled=not inc_max_depth)

    inc_subsample = st.checkbox("Include subsample", value=True)
    st.text_input("subsample list", key="gs_subsample", disabled=not inc_subsample)

with right_cols:
    inc_gamma = st.checkbox("Include gamma", value=False)
    st.text_input("gamma list", key="gs_gamma", disabled=not inc_gamma)

    inc_max_delta_step = st.checkbox("Include max_delta_step", value=False)
    st.text_input("max_delta_step list", key="gs_max_delta_step", disabled=not inc_max_delta_step)

    inc_colsample = st.checkbox("Include colsample_* (bytree/bylevel/bynode)", value=False)
    st.text_input("colsample list", key="gs_colsample", disabled=not inc_colsample)

search_type = st.radio("Search type", ["Grid", "Randomized"], horizontal=True)
if search_type == "Randomized":
    n_iter = st.slider("n_iter (RandomizedSearchCV)", 2, 50, 10)

# Build param grid only from the selected options
param_grid = {}
if inc_n_estimators:
    vals = parse_values(st.session_state.get("gs_n_estimators", ""))
    if vals: param_grid['n_estimators'] = vals
if inc_learning_rate:
    vals = parse_values(st.session_state.get("gs_learning_rate", ""))
    if vals: param_grid['learning_rate'] = vals
if inc_max_depth:
    vals = parse_values(st.session_state.get("gs_max_depth", ""))
    if vals: param_grid['max_depth'] = vals
if inc_subsample:
    vals = parse_values(st.session_state.get("gs_subsample", ""))
    if vals: param_grid['subsample'] = vals
if inc_gamma:
    vals = parse_values(st.session_state.get("gs_gamma", ""))
    if vals: param_grid['gamma'] = vals
if inc_max_delta_step:
    vals = parse_values(st.session_state.get("gs_max_delta_step", ""))
    if vals: param_grid['max_delta_step'] = vals
if inc_colsample:
    vals = parse_values(st.session_state.get("gs_colsample", ""))
    if vals:
        param_grid['colsample_bytree'] = vals
        param_grid['colsample_bylevel'] = vals
        param_grid['colsample_bynode'] = vals

# Runner
run_sel = st.button("▶️ Run selected search")
if run_sel:
    if not use_xgb or not XGB_AVAILABLE:
        st.error("Grid/Randomized search requires XGBoost.")
    elif not param_grid:
        st.warning("Select at least one hyperparameter.")
    else:
        model_base = make_model()
        kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        if search_type == "Randomized":
            # limit n_iter to total combos
            total = 1
            for v in param_grid.values():
                total *= max(1, len(v))
            n_iter_eff = min(n_iter, total)
            search = RandomizedSearchCV(model_base, param_grid, cv=kfold, n_jobs=-1,
                                        random_state=random_state, scoring='recall', n_iter=n_iter_eff)
        else:
            search = GridSearchCV(model_base, param_grid, cv=kfold, n_jobs=-1, scoring='recall')
        search.fit(X_gs, y_gs)
        st.write("Best params:", search.best_params_)
        st.write(f"Best score: {search.best_score_:.5f}")
        st.write("All scores:", search.cv_results_['mean_test_score'])

        # Show tidy table as well
        cv_df = pd.DataFrame(search.cv_results_)
        cols_to_show = [c for c in cv_df.columns if c.startswith('param_') or c in ['mean_test_score','std_test_score','rank_test_score']]
        st.dataframe(cv_df[cols_to_show].sort_values('rank_test_score'), use_container_width=True)

# -------------------------------
# Final-model style evaluation (train on selected scope, evaluate on ALL)
# -------------------------------
st.header("✅ Final-Model Style Evaluation")
st.caption("Train on selected scope (Short/Loaded/All) then evaluate on **All rows** (X_all vs y_all), mirroring your `final_model` function.")

if st.button("Run final-model evaluation (on ALL rows)"):
    model = make_model()
    model.fit(X_used, y_used)
    y_pred_all = model.predict(X_all)
    rec_all = recall_score(y_all, y_pred_all, zero_division=0)
    st.write("**Recall (on ALL):**", f"{rec_all:.4f}")

    st.subheader("Confusion matrix (ALL)")
    cm = confusion_matrix(y_all, y_pred_all, labels=[0,1])
    fig, ax = plt.subplots(figsize=(4,3))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix — ALL")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_xticklabels(["0","1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["0","1"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    st.pyplot(fig)

    st.subheader("Classification report (ALL)")
    st.dataframe(classification_report_df(y_all, y_pred_all), use_container_width=True)

    # Downloadables
    out = X_all.copy()
    out["y_true"] = y_all
    out["y_pred"] = y_pred_all
    st.download_button("⬇️ Download ALL predictions (CSV)", data=out.to_csv(index=False).encode("utf-8"), file_name="predictions_all.csv", mime="text/csv")

# -------------------------------
# Save / load trained model
# -------------------------------
save_col, load_col = st.columns(2)
with save_col:
    if st.button("💾 Save current base model"):
        model = make_model()
        model.fit(X_train, y_train)
        buf = io.BytesIO()
        joblib.dump(model, buf)
        st.download_button("Download model (.joblib)", data=buf.getvalue(), file_name="exoplanet_model.joblib", mime="application/octet-stream")
with load_col:
    upm = st.file_uploader("Load .joblib model for scoring", type=["joblib"], key="model_upl")
    if upm is not None:
        try:
            loaded_model = joblib.load(upm)
            preds = loaded_model.predict(X_test)
            st.write("Loaded model test accuracy:", f"{accuracy_score(y_test, preds):.3f}")
        except Exception as e:
            st.error(f"Could not load model: {e}")
