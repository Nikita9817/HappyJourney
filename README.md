# HappyJourney
# streamlit_app.py
# Missed Connection Risk Predictor – a fresh airline ML project for Streamlit
# Idea: Predict probability that an itinerary will result in a missed connection, using flight, airport, and connection features.
# This app lets you: upload data, map columns, train a model, evaluate it, and make predictions on new rows.

import io
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Missed Connection Risk Predictor", page_icon="✈️", layout="wide")
st.title("✈️ Missed Connection Risk Predictor")
st.caption("Fresh airline ML project: predict the risk a passenger will miss a connection. Train, evaluate, and deploy – all in one Streamlit app.")

with st.expander("About this idea (why it's fresh)", expanded=False):
    st.markdown(
        """
        **Why this is fresh:** Traditional delay models predict departure or arrival delays. This app goes a step further: it estimates the
        risk a **connecting passenger** will **miss their onward flight**, combining layover buffers, inbound delays, gate changes, airport hub congestion,
        and operational flags (e.g., aircraft swaps). Airlines can use it for rebooking prioritization, proactive alerts, or smarter minimum-connection-time rules.
        """
    )

# ========================
# 1) Data Upload
# ========================
st.header("1) Upload your historical connection dataset")

st.markdown(
    """
    **CSV requirements** (you can have more columns):
    - One **target** column: 1 = missed connection, 0 = made connection.
    - Typical **numeric features**: inbound_arr_delay, layover_minutes, distance_next_leg, scheduled_turnaround, taxi_in, taxi_out, 
      weather_severity_index, gate_distance_m, prev_gate_change_count.
    - Typical **categorical features**: airline, origin, connection_airport, destination, weekday, holiday_flag, aircraft_type, aircraft_swap_flag.

    *Don't have all these? No problem. The app lets you map whatever you have.*
    """
)

uploaded = st.file_uploader("Upload CSV", type=["csv"]) 

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        df = pd.read_csv(uploaded, encoding_errors="ignore")

    st.subheader("Preview")
    st.dataframe(df.head(20))

    with st.expander("Dataset shape & basic info"):
        buf = io.StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())
        st.write({"rows": df.shape[0], "columns": df.shape[1]})

    # ========================
    # 2) Column Mapping
    # ========================
    st.header("2) Map your columns")

    target_col = st.selectbox("Select the target column (1 = missed, 0 = made)", options=[None] + list(df.columns), index=0)

    # Heuristic suggestions
    numeric_guess = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    categorical_guess = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    num_cols = st.multiselect("Numeric feature columns", options=list(df.columns), default=numeric_guess)
    cat_cols = st.multiselect("Categorical feature columns", options=list(df.columns), default=categorical_guess)

    # Remove target from features if accidentally included
    num_cols = [c for c in num_cols if c != target_col]
    cat_cols = [c for c in cat_cols if c != target_col]

    # ========================
    # 3) Train / Eval
    # ========================
    st.header("3) Train the model")

    col_left, col_right = st.columns([2,1])

    with col_right:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42)
        max_depth = st.slider("Max depth (HGB)", 2, 16, 8)
        learning_rate = st.select_slider("Learning rate", options=[0.01, 0.02, 0.05, 0.1, 0.2], value=0.1)
        max_iter = st.slider("Max iter (trees)", 100, 1000, 300, 50)
        l2 = st.select_slider("L2 regularization", options=[0.0, 0.1, 0.5, 1.0], value=0.1)

    with col_left:
        if st.button("🚀 Train model", type="primary", use_container_width=True, disabled=(target_col is None)):
            if target_col is None:
                st.error("Please select a target column first.")
            else:
                # Drop rows with missing target
                work = df.dropna(subset=[target_col]).copy()

                y = work[target_col].astype(int)
                X = work[num_cols + cat_cols]

                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=int(random_state), stratify=y
                )

                # Preprocess
                preprocess = ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(with_mean=False), num_cols),
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
                    ], remainder="drop"
                )

                clf = HistGradientBoostingClassifier(
                    max_depth=int(max_depth),
                    learning_rate=float(learning_rate),
                    max_iter=int(max_iter),
                    l2_regularization=float(l2),
                    random_state=int(random_state)
                )

                pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])

                with st.spinner("Training..."):
                    start = time.time()
                    pipe.fit(X_train, y_train)
                    train_time = time.time() - start

                st.success(f"Model trained in {train_time:.2f}s")

                # Evaluate
                y_proba = pipe.predict_proba(X_test)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)

                roc = roc_auc_score(y_test, y_proba)
                f1 = f1_score(y_test, y_pred)
                pr, rc, th = precision_recall_curve(y_test, y_proba)
                pr_auc = auc(rc, pr)

                cmat = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("ROC-AUC", f"{roc:.3f}")
                m2.metric("PR-AUC", f"{pr_auc:.3f}")
                m3.metric("F1 score", f"{f1:.3f}")

                # PR Curve plot
                fig1 = plt.figure()
                plt.plot(rc, pr)
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.title("Precision-Recall Curve")
                st.pyplot(fig1, clear_figure=True)

                # Confusion matrix
                fig2 = plt.figure()
                im = plt.imshow(cmat, interpolation="nearest")
                plt.title("Confusion Matrix (thr=0.5)")
                plt.colorbar(im)
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ["Made", "Missed"]) 
                plt.yticks(tick_marks, ["Made", "Missed"]) 
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                for (i, j), val in np.ndenumerate(cmat):
                    plt.text(j, i, int(val), ha='center', va='center')
                st.pyplot(fig2, clear_figure=True)

                with st.expander("Classification report"):
                    st.json(report)

                # Save model to session state
                st.session_state["trained_pipe"] = pipe
                st.session_state["feature_cols_num"] = num_cols
                st.session_state["feature_cols_cat"] = cat_cols
                st.session_state["target_col"] = target_col

    # ========================
    # 4) Threshold tuning & export
    # ========================
    st.header("4) Predict on new data / tune threshold / export model")

    pipe = st.session_state.get("trained_pipe")
    if pipe is None:
        st.info("Train a model first to enable predictions and export.")
    else:
        st.markdown("**Upload new rows (same columns as training features) to get risk scores:**")
        newfile = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred_csv")

        thr = st.slider("Decision threshold (probability of miss)", 0.05, 0.95, 0.5, 0.01)

        if newfile is not None:
            newdf = pd.read_csv(newfile)
            st.write("Preview of inputs:")
            st.dataframe(newdf.head(10))

            # Only keep mapped feature columns
            feats = st.session_state["feature_cols_num"] + st.session_state["feature_cols_cat"]
            missing = [c for c in feats if c not in newdf.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                probs = pipe.predict_proba(newdf[feats])[:, 1]
                preds = (probs >= thr).astype(int)
                out = newdf.copy()
                out["miss_risk_proba"] = probs
                out["miss_pred"] = preds

                st.subheader("Predictions")
                st.dataframe(out.head(50))

                # Download predictions
                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", csv, file_name="missed_connection_predictions.csv", mime="text/csv")

        # Export model
        if st.button("💾 Export trained model (joblib)"):
            import joblib
            bytes_io = io.BytesIO()
            joblib.dump(pipe, bytes_io)
            st.download_button(
                label="Download model.joblib",
                data=bytes_io.getvalue(),
                file_name="missed_connection_model.joblib",
                mime="application/octet-stream"
            )

else:
    st.info("Upload a CSV to begin. If you don’t have a labeled target yet, you can still explore your columns and plan features.")


st.markdown("---")
st.subheader("Feature engineering ideas (optional but powerful)")
st.markdown(
    """
    - **Layover tightness**: `layover_minutes - published_MCT(origin, connection_airport)` (negative values are risky).
    - **Inbound variance**: rolling std of inbound arrival delay for the last N days on the same flight/route.
    - **Gate logistics**: estimated walking time between gates, or historical average gate changes at the connection airport/time of day.
    - **Network pressure**: departures per 15 minutes window at connection airport (proxy for congestion).
    - **Weather indices**: discretized visibility, wind, precipitation categories for origin/connection airport.
    - **Crew/aircraft connectivity**: flags indicating if inbound aircraft turns for the onward leg, or aircraft swap probability.
    - **Calendar effects**: weekday, holiday, peak season.
    """
)
