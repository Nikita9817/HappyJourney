

**app.py*
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

st.title("✈️ Missed Connection Risk Predictor")

st.write("Upload your dataset to train a model predicting missed flight connections.")

# File upload
data_file = st.file_uploader("Upload CSV", type=["csv"])
if data_file:
    df = pd.read_csv(data_file)
    st.write("Preview:", df.head())
    
    target_col = st.selectbox("Select target column (1=missed, 0=made)", df.columns)
    feature_cols = st.multiselect("Select feature columns", [c for c in df.columns if c != target_col])
    
    if st.button("Train Model"):
        X = df[feature_cols]
        y = df[target_col]
        X = pd.get_dummies(X, drop_first=True)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = HistGradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        st.write(f"ROC-AUC: {auc:.3f}")
        st.write(f"F1 Score: {f1:.3f}")
        
        # PR curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        st.pyplot(fig)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        st.write(cm)
        
        joblib.dump(model, "missed_connection_model.joblib")
        st.success("Model trained and saved!")

# Prediction on new data
st.subheader("Predict on new data")
new_file = st.file_uploader("Upload new data CSV for prediction", type=["csv"], key="pred")
if new_file:
    model = joblib.load("missed_connection_model.joblib")
    new_df = pd.read_csv(new_file)
    X_new = pd.get_dummies(new_df, drop_first=True)
    
    # Align columns with training data
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[model_features]
    
    preds = model.predict_proba(X_new)[:, 1]
    new_df['Missed_Connection_Risk'] = preds
    st.write(new_df)
    new_df.to_csv("predictions.csv", index=False)
    st.download_button("Download Predictions", data=new_df.to_csv(index=False), file_name="predictions.csv")








