import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Iris ML Classifier Evaluation", layout="wide")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.pkl')
    models = {
        "Logistic Regression": joblib.load('model/logistic_regression.pkl'),
        "Decision Tree": joblib.load('model/decision_tree.pkl'),
        "KNN": joblib.load('model/knn.pkl'),
        "Naive Bayes": joblib.load('model/naive_bayes.pkl'),
        "Support Vector Machine": joblib.load('model/svm.pkl')
    }
    return scaler, models

scaler, models = load_artifacts()

st.title("Iris Species Classification Dashboard")
st.markdown("Interactive Streamlit web application evaluating multiple classification models.")

# Sidebar Dataset Upload
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
elif os.path.exists("test_data.csv"):
    df_test = pd.read_csv("test_data.csv")
else:
    st.info("Please upload a test dataset CSV file to proceed.")
    st.stop()

target_col = 'target' if 'target' in df_test.columns else 'Species'
X_test = df_test.drop(columns=[target_col])
if 'Id' in X_test.columns:
    X_test = X_test.drop(columns=['Id'])
y_test = df_test[target_col]

# Model Selection Dropdown
model_name = st.sidebar.selectbox("Select Classifier", list(models.keys()))
selected_model = models[model_name]

# Inference
X_test_scaled = scaler.transform(X_test)
y_pred = selected_model.predict(X_test_scaled)
y_prob = selected_model.predict_proba(X_test_scaled) if hasattr(selected_model, "predict_proba") else None

# Metrics Calculation
acc = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr') if y_prob is not None else 0.0
except Exception:
    auc = 0.0
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

# Display Metrics
st.subheader(f"Evaluation Metrics: {model_name}")
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{acc:.4f}")
col2.metric("AUC Score", f"{auc:.4f}")
col3.metric("Precision", f"{prec:.4f}")
col4.metric("Recall", f"{rec:.4f}")
col5.metric("F1 Score", f"{f1:.4f}")
col6.metric("MCC Score", f"{mcc:.4f}")

st.divider()

# Confusion Matrix & Classification Report
col_cm, col_rep = st.columns([1, 1])

with col_cm:
    st.subheader("Confusion Matrix")
    labels = sorted(list(y_test.unique()))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

with col_rep:
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report_dict).transpose())