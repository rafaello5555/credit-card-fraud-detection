import streamlit as st
import pandas as pd
from eda import run_eda
from data_preprocessing import preprocess_data
from model import train_decision_tree, train_svm

# Streamlit page
st.set_page_config(page_title="Credit Card Fraud Detection")
st.title("💳 Credit Card Fraud Detection Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")
data = load_data()

# EDA
st.subheader("Dataset Overview")
eda_stats = run_eda(data)
st.write(f"Number of transactions: {eda_stats['num_rows']}")
st.write(f"Number of variables: {eda_stats['num_columns']}")
st.dataframe(eda_stats['head'])
st.pyplot(eda_stats['pie_chart'])
st.pyplot(eda_stats['histogram'])

# Preprocessing
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train Decision Tree
st.subheader("Decision Tree Classifier")
dt_clf, dt_time, dt_roc_auc = train_decision_tree(X_train, y_train, X_test, y_test)
st.write(f"Training time: {dt_time:.3f} seconds")
st.write(f"ROC-AUC score: {dt_roc_auc:.3f}")

# Train SVM
st.subheader("Linear SVM Classifier")
svm_clf, svm_time, svm_roc_auc, svm_hinge = train_svm(X_train, y_train, X_test, y_test)
st.write(f"Training time: {svm_time:.3f} seconds")
st.write(f"ROC-AUC score: {svm_roc_auc:.3f}")
st.write(f"Hinge loss: {svm_hinge:.3f}")
