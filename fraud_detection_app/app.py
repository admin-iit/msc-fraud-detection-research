import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔍",
    layout="wide"
)

# ── Load model and samples ───────────────────────────────────
@st.cache_resource
def load_model():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base_dir, 'best_model_xgb.pkl'))

@st.cache_data
def load_samples():
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base_dir, 'sample_transactions.csv'))

model = load_model()
samples_df = load_samples()

# ── Results data (from your experiments) ─────────────────────
results_data = {
    'Configuration': [
        'LR + Undersampling', 'LR + SMOTE', 'LR + Cost-Sensitive',
        'RF + Undersampling', 'RF + SMOTE', 'RF + Cost-Sensitive',
        'XGBoost + Undersampling', 'XGBoost + SMOTE', 'XGBoost + Cost-Sensitive'
    ],
    'F1': [0.0926, 0.1109, 0.1214, 0.1140, 0.8574, 0.8510,
            0.0354, 0.6319, 0.8687],
    'AUROC': [0.9769, 0.9779, 0.9795, 0.9775, 0.9731, 0.9508,
               0.9755, 0.9660, 0.9782],
    'AUPRC': [0.7107, 0.7300, 0.7292, 0.7398, 0.8538, 0.8476,
               0.7338, 0.8405, 0.8612]
}
results_df = pd.DataFrame(results_data)

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Dashboard",
    "🔍 Live Demo",
    "ℹ️ About"
])

# ════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Research Dashboard")
    st.markdown("**Integrating Class Imbalance Techniques and Explainable AI for Enhanced Credit Card Fraud Detection**")
    st.markdown("---")

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Size", "284,807", "transactions")
    col2.metric("Fraud Cases", "492", "0.17% of total")
    col3.metric("Best F1 Score", "0.869", "XGBoost + Cost-Sensitive")
    col4.metric("Best AUPRC", "0.885", "After tuning")

    st.markdown("---")

    # Class distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['Legitimate', 'Fraud'], [284315, 492],
               color=['#2196F3', '#F44336'])
        ax.set_ylabel('Count')
        ax.set_title('Transaction Class Distribution')
        for i, v in enumerate([284315, 492]):
            ax.text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Class Imbalance")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie([284315, 492],
               labels=['Legitimate\n99.83%', 'Fraud\n0.17%'],
               colors=['#2196F3', '#F44336'],
               startangle=90,
               explode=(0, 0.1))
        ax.set_title('Fraud vs Legitimate Transactions')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # F1 Score comparison
    st.subheader("F1 Score Comparison — All 9 Configurations")
    colors = ['#F44336' if 'XGBoost + Cost' in c else
              '#4CAF50' if f1 > 0.5 else '#9E9E9E'
              for c, f1 in zip(results_df['Configuration'], results_df['F1'])]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.barh(results_df['Configuration'], results_df['F1'], color=colors)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score by Configuration')
    ax.set_xlim(0, 1.0)
    for bar, val in zip(bars, results_df['F1']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # AUROC and AUPRC
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AUROC Comparison")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(results_df['Configuration'], results_df['AUROC'],
                color='#3F51B5')
        ax.set_xlabel('AUROC')
        ax.set_xlim(0.9, 1.0)
        ax.set_title('AUROC by Configuration')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("AUPRC Comparison")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(results_df['Configuration'], results_df['AUPRC'],
                color='#009688')
        ax.set_xlabel('AUPRC')
        ax.set_xlim(0.6, 1.0)
        ax.set_title('AUPRC by Configuration')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # Confusion matrix
    st.subheader("Confusion Matrix — Tuned XGBoost + Cost-Sensitive")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("""
        | | Predicted Legit | Predicted Fraud |
        |---|---|---|
        | **Actual Legit** | 56,852 ✅ | 12 ❌ |
        | **Actual Fraud** | 16 ❌ | 82 ✅ |
        """)
        st.info("82 out of 98 fraud cases correctly caught")
        st.success("Only 12 false alarms out of 56,864 legitimate transactions")

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = np.array([[56852, 12], [16, 82]])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Legitimate', 'Fraud'])
        ax.set_yticklabels(['Legitimate', 'Fraud'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                       fontsize=14, color='white' if cm[i,j] > 30000 else 'black',
                       fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 2: LIVE DEMO
# ════════════════════════════════════════════════════════════
elif page == "🔍 Live Demo":
    st.title("🔍 Live Fraud Detection Demo")
    st.markdown("Select a transaction to see the model prediction and SHAP explanation.")
    st.markdown("---")

    # Build dropdown options
    options = []
    for i, row in samples_df.iterrows():
        label = row.get('label', 'Unknown')
        amount = abs(row['Amount'])
        options.append(f"Transaction {i+1} — {label} (Amount: ${amount:.2f})")

    selected = st.selectbox("Select a transaction:", options)
    selected_idx = options.index(selected)
    transaction = samples_df.iloc[selected_idx]

    st.markdown("---")

    # Show transaction details
    col1, col2, col3 = st.columns(3)
    col1.metric("Transaction Amount", f"${abs(transaction['Amount']):.2f}")
    col2.metric("Transaction Time", f"{abs(transaction['Time']):.0f}s")
    actual_label = transaction.get('label', 'Unknown')
    col3.metric("Actual Label", actual_label)

    st.markdown("---")

    # Run prediction
    if st.button("🔍 Analyse Transaction", type="primary"):
        feature_cols = [c for c in samples_df.columns if c != 'label']
        X_input = transaction[feature_cols].values.reshape(1, -1)

        prob = model.predict_proba(X_input)[0][1]
        pred = "🚨 FRAUD" if prob > 0.5 else "✅ LEGITIMATE"
        color = "red" if prob > 0.5 else "green"

        st.markdown(f"### Prediction: :{color}[{pred}]")

        col1, col2 = st.columns(2)
        col1.metric("Fraud Probability", f"{prob:.1%}")
        col2.metric("Confidence", f"{max(prob, 1-prob):.1%}")

        st.markdown("---")
        st.subheader("SHAP Explanation — Why did the model make this decision?")

        with st.spinner("Computing SHAP explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(
                pd.DataFrame(X_input, columns=feature_cols))

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_input[0],
                    feature_names=feature_cols
                ), show=False
            )
            st.pyplot(fig)
            plt.close()

        st.info("Red bars push toward FRAUD. Blue bars push toward LEGITIMATE. The longer the bar, the stronger the influence.")

# ════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Research")
    st.markdown("""
    ## Integrating Class Imbalance Techniques and Explainable AI for Enhanced Credit Card Fraud Detection

    **Student:** Thiwanka Sandaruwan
    **Supervisor:** Mr. Cassim Farook
    **Programme:** MSc Big Data Analytics
    **University:** Robert Gordon University

    ---

    ### Research Overview
    This research systematically compares three class imbalance handling techniques across
    three machine learning classifiers for credit card fraud detection, and applies SHAP-based
    explainability to understand what drives each model's decisions.

    ### Models Evaluated
    - Logistic Regression (baseline)
    - Random Forest (ensemble)
    - XGBoost (gradient boosting)

    ### Imbalance Techniques
    - Random Undersampling
    - SMOTE (Synthetic Minority Over-sampling)
    - Cost-Sensitive Learning

    ### Best Result
    **XGBoost + Cost-Sensitive Learning** achieved:
    - F1 Score: 0.869
    - AUROC: 0.982
    - AUPRC: 0.885
    - Precision: 0.908
    - Recall: 0.827

    ### Dataset
    European Credit Card Fraud Detection Dataset (Kaggle)
    - 284,807 transactions
    - 492 fraud cases (0.17%)
    - 578:1 class imbalance ratio

    ---
    *MSc Research Project 2025-2026*
    """)
