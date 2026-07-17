import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os

st.set_page_config(
    page_title="Fraud Detection Research",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1f3a 0%, #2d3561 100%); }
    [data-testid="stSidebar"] * { color: white !important; }
    .metric-card { background: white; border-radius: 16px; padding: 20px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; text-align: center; }
    .metric-value { font-size: 32px; font-weight: 700; color: #1a1f3a; margin: 8px 0 4px; }
    .metric-label { font-size: 13px; color: #6b7280; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-sub { font-size: 12px; color: #9ca3af; margin-top: 4px; }
    .section-header { font-size: 20px; font-weight: 700; color: #1a1f3a; margin: 24px 0 16px; padding-bottom: 10px; border-bottom: 2px solid #e5e7eb; }
    .page-title { font-size: 28px; font-weight: 800; color: #1a1f3a; margin-bottom: 4px; }
    .page-subtitle { font-size: 15px; color: #6b7280; margin-bottom: 24px; }
    .result-fraud { background: #fef2f2; border: 2px solid #ef4444; border-radius: 16px; padding: 24px; text-align: center; }
    .result-legit { background: #f0fdf4; border: 2px solid #22c55e; border-radius: 16px; padding: 24px; text-align: center; }
    .result-title-fraud { font-size: 28px; font-weight: 800; color: #dc2626; }
    .result-title-legit { font-size: 28px; font-weight: 800; color: #16a34a; }
    .info-banner { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 12px; padding: 16px 20px; font-size: 14px; color: #1d4ed8; margin: 16px 0; }
    .transaction-card { background: white; border-radius: 16px; padding: 20px 24px; box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #f0f0f0; margin-bottom: 20px; }
    .badge-fraud { background: #fef2f2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #fecaca; }
    .badge-legit { background: #f0fdf4; color: #16a34a; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; border: 1px solid #bbf7d0; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    .stButton > button { background: linear-gradient(135deg, #1a1f3a 0%, #2d3561 100%); color: white; border: none; border-radius: 12px; padding: 14px 32px; font-size: 16px; font-weight: 600; width: 100%; }
</style>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(base_dir, 'best_model_xgb.pkl'))

@st.cache_data
def load_samples():
    return pd.read_csv(os.path.join(base_dir, 'sample_transactions.csv'))

model = load_model()
samples_df = load_samples()
feature_cols = [c for c in samples_df.columns if c not in ['label', 'Class']]

results_data = {
    'Configuration': ['LR + Undersampling','LR + SMOTE','LR + Cost-Sensitive','RF + Undersampling','RF + SMOTE','RF + Cost-Sensitive','XGBoost + Undersampling','XGBoost + SMOTE','XGBoost + Cost-Sensitive'],
    'F1': [0.0926,0.1109,0.1214,0.1140,0.8574,0.8510,0.0354,0.6319,0.8687],
    'AUROC': [0.9769,0.9779,0.9795,0.9775,0.9731,0.9508,0.9755,0.9660,0.9782],
    'AUPRC': [0.7107,0.7300,0.7292,0.7398,0.8538,0.8476,0.7338,0.8405,0.8612]
}
results_df = pd.DataFrame(results_data)

with st.sidebar:
    st.markdown("""
    <div style="padding:20px 0 30px;">
        <div style="font-size:22px;font-weight:800;color:white;margin-bottom:4px;">🛡️ FraudGuard AI</div>
        <div style="font-size:12px;color:rgba(255,255,255,0.5);margin-bottom:30px;">MSc Research — RGU 2026</div>
    </div>""", unsafe_allow_html=True)
    page = st.radio("Navigation", options=["Overview","Model Performance","Live Detection","Research Info"], label_visibility="hidden")
    st.markdown("""
    <div style="margin-top:40px;">
        <div style="font-size:11px;color:rgba(255,255,255,0.35);line-height:1.6;">
            Thiwanka Sandaruwan<br>MSc Big Data Analytics<br>Robert Gordon University
        </div>
    </div>""", unsafe_allow_html=True)

def clean_chart():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.tick_params(colors='#6b7280')
    return fig, ax

if page == "Overview":
    st.markdown('<div class="page-title">Research overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Integrating Class Imbalance Techniques and Explainable AI for Enhanced Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-label">Dataset size</div><div class="metric-value">284,807</div><div class="metric-sub">transactions</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-label">Fraud cases</div><div class="metric-value" style="color:#dc2626">492</div><div class="metric-sub">0.17% of total</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-label">Best F1 score</div><div class="metric-value" style="color:#16a34a">0.869</div><div class="metric-sub">XGBoost + Cost-Sensitive</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-label">Best AUPRC</div><div class="metric-value" style="color:#2563eb">0.885</div><div class="metric-sub">After tuning</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Class distribution</div>', unsafe_allow_html=True)
        fig, ax = clean_chart()
        fig.set_size_inches(6,4)
        bars = ax.bar(['Legitimate','Fraud'],[284315,492],color=['#2563eb','#dc2626'],width=0.5,edgecolor='none')
        ax.set_ylabel('Count',fontsize=12,color='#6b7280')
        for bar,v in zip(bars,[284315,492]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1500, f'{v:,}', ha='center', fontsize=11, fontweight='600', color='#1a1f3a')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        st.markdown('<div class="section-header">Fraud vs legitimate ratio</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('white')
        wedges,texts,autotexts = ax.pie([284315,492],labels=['Legitimate','Fraud'],colors=['#2563eb','#dc2626'],autopct='%1.2f%%',startangle=90,explode=(0,0.08),wedgeprops={'edgecolor':'white','linewidth':2})
        for t in texts: t.set_fontsize(12)
        for a in autotexts: a.set_fontsize(11); a.set_color('white'); a.set_fontweight('bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="section-header">Confusion matrix — tuned XGBoost + cost-sensitive</div>', unsafe_allow_html=True)
    col1,col2 = st.columns([1,2])
    with col1:
        st.markdown("""<div class="transaction-card">
        <table style="width:100%;font-size:14px;border-collapse:collapse;">
        <tr><th style="padding:10px;color:#6b7280;font-weight:500;"></th><th style="padding:10px;color:#6b7280;font-weight:500;text-align:center;">Predicted legit</th><th style="padding:10px;color:#6b7280;font-weight:500;text-align:center;">Predicted fraud</th></tr>
        <tr><td style="padding:10px;font-weight:600;color:#1a1f3a;">Actual legit</td><td style="padding:10px;text-align:center;color:#16a34a;font-weight:700;font-size:18px;">56,852 ✅</td><td style="padding:10px;text-align:center;color:#dc2626;font-weight:700;font-size:18px;">12 ❌</td></tr>
        <tr><td style="padding:10px;font-weight:600;color:#1a1f3a;">Actual fraud</td><td style="padding:10px;text-align:center;color:#dc2626;font-weight:700;font-size:18px;">16 ❌</td><td style="padding:10px;text-align:center;color:#16a34a;font-weight:700;font-size:18px;">82 ✅</td></tr>
        </table>
        <div style="margin-top:16px;padding:12px;background:#f0fdf4;border-radius:8px;font-size:13px;color:#16a34a;">✅ 82 of 98 fraud cases correctly identified</div>
        <div style="margin-top:8px;padding:12px;background:#eff6ff;border-radius:8px;font-size:13px;color:#1d4ed8;">🎯 Only 12 false alarms out of 56,864 legitimate transactions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6,4)); fig.patch.set_facecolor('white')
        cm = np.array([[56852,12],[16,82]])
        ax.imshow(cm,cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Legitimate','Fraud'],fontsize=12)
        ax.set_yticklabels(['Legitimate','Fraud'],fontsize=12)
        ax.set_xlabel('Predicted',fontsize=12,color='#6b7280')
        ax.set_ylabel('Actual',fontsize=12,color='#6b7280')
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{cm[i,j]:,}',ha='center',va='center',fontsize=16,color='white' if cm[i,j]>30000 else '#1a1f3a',fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

elif page == "Model Performance":
    st.markdown('<div class="page-title">Model performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comparison across all 9 experimental configurations (3 models × 3 imbalance techniques)</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">F1 score comparison</div>', unsafe_allow_html=True)
    colors_f1 = ['#dc2626' if 'XGBoost + Cost' in c else '#16a34a' if f1>0.5 else '#d1d5db' for c,f1 in zip(results_df['Configuration'],results_df['F1'])]
    fig, ax = clean_chart(); fig.set_size_inches(12,5)
    bars = ax.barh(results_df['Configuration'],results_df['F1'],color=colors_f1,edgecolor='none',height=0.6)
    ax.set_xlabel('F1 Score',fontsize=12,color='#6b7280'); ax.set_xlim(0,1.05)
    ax.tick_params(labelsize=11)
    for bar,val in zip(bars,results_df['F1']):
        ax.text(val+0.01,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=10,color='#1a1f3a',fontweight='500')
    plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="info-banner">🏆 <strong>Best configuration:</strong> XGBoost + Cost-Sensitive Learning achieved F1 = 0.8687, AUROC = 0.9782, AUPRC = 0.8612 — highlighted in red above.</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">AUROC comparison</div>', unsafe_allow_html=True)
        fig, ax = clean_chart(); fig.set_size_inches(6,5)
        ax.barh(results_df['Configuration'],results_df['AUROC'],color='#2563eb',edgecolor='none',height=0.6)
        ax.set_xlabel('AUROC',fontsize=11,color='#6b7280'); ax.set_xlim(0.9,1.0); ax.tick_params(labelsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col2:
        st.markdown('<div class="section-header">AUPRC comparison</div>', unsafe_allow_html=True)
        fig, ax = clean_chart(); fig.set_size_inches(6,5)
        ax.barh(results_df['Configuration'],results_df['AUPRC'],color='#7c3aed',edgecolor='none',height=0.6)
        ax.set_xlabel('AUPRC',fontsize=11,color='#6b7280'); ax.set_xlim(0.6,1.0); ax.tick_params(labelsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    st.markdown('<div class="section-header">Full results table</div>', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

elif page == "Live Detection":
    st.markdown('<div class="page-title">Live fraud detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Select a transaction to analyse using the trained XGBoost model with SHAP explainability</div>', unsafe_allow_html=True)
    options = []
    for i, row in samples_df.iterrows():
        label = row.get('label','Unknown')
        amount = abs(row['Amount'])
        icon = "🚨" if label=='Fraud' else "✅"
        options.append(f"{icon} Transaction {i+1} — {label}  |  Amount: ${amount:.2f}")
    st.markdown('<div class="section-header">Select a transaction</div>', unsafe_allow_html=True)
    selected = st.selectbox("", options, label_visibility="hidden")
    selected_idx = options.index(selected)
    transaction = samples_df.iloc[selected_idx]
    actual_label = transaction.get('label','Unknown')
    st.markdown('<div class="section-header">Transaction details</div>', unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Amount</div><div class="metric-value">${abs(transaction["Amount"]):.2f}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{abs(transaction["Time"]):.0f}s</div></div>', unsafe_allow_html=True)
    with col3:
        badge_class = "badge-fraud" if actual_label=="Fraud" else "badge-legit"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Actual label</div><div style="margin-top:12px;"><span class="{badge_class}">{actual_label}</span></div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Analyse this transaction"):
        X_input = pd.DataFrame([transaction[feature_cols].values], columns=feature_cols)
        prob = model.predict_proba(X_input)[0][1]
        is_fraud = prob > 0.5
        st.markdown('<div class="section-header">Prediction result</div>', unsafe_allow_html=True)
        if is_fraud:
            st.markdown(f'<div class="result-fraud"><div class="result-title-fraud">🚨 Fraud detected</div><div style="font-size:18px;color:#6b7280;margin-top:8px;">Fraud probability: <strong style="color:#dc2626">{prob:.1%}</strong> &nbsp;|&nbsp; Confidence: <strong style="color:#dc2626">{max(prob,1-prob):.1%}</strong></div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-legit"><div class="result-title-legit">✅ Legitimate transaction</div><div style="font-size:18px;color:#6b7280;margin-top:8px;">Fraud probability: <strong style="color:#16a34a">{prob:.1%}</strong> &nbsp;|&nbsp; Confidence: <strong style="color:#16a34a">{max(prob,1-prob):.1%}</strong></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">SHAP explanation — why did the model decide this?</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-banner">🔴 Red bars push toward <strong>fraud</strong>. 🔵 Blue bars push toward <strong>legitimate</strong>. The longer the bar, the stronger the influence.</div>', unsafe_allow_html=True)
        with st.spinner("Computing SHAP explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_input)
            fig, ax = plt.subplots(figsize=(10,6)); fig.patch.set_facecolor('white')
            shap.waterfall_plot(shap.Explanation(values=shap_values[0],base_values=explainer.expected_value,data=X_input.values[0],feature_names=feature_cols),show=False)
            st.pyplot(fig); plt.close()

elif page == "Research Info":
    st.markdown('<div class="page-title">Research information</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">About this MSc research project</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="transaction-card">
        <div style="font-size:16px;font-weight:700;color:#1a1f3a;margin-bottom:16px;">Project details</div>
        <table style="width:100%;font-size:14px;">
        <tr><td style="color:#6b7280;padding:6px 0;">Student</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">Thiwanka Sandaruwan</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Supervisor</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">Mr. Cassim Farook</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Programme</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">MSc Big Data Analytics</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">University</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">Robert Gordon University</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Year</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">2025–2026</td></tr>
        </table></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="transaction-card" style="margin-top:16px;">
        <div style="font-size:16px;font-weight:700;color:#1a1f3a;margin-bottom:16px;">Dataset</div>
        <table style="width:100%;font-size:14px;">
        <tr><td style="color:#6b7280;padding:6px 0;">Source</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">Kaggle (European CC Fraud)</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Transactions</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">284,807</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Fraud cases</td><td style="font-weight:600;color:#dc2626;padding:6px 0;">492 (0.17%)</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Imbalance ratio</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">578:1</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Features</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">30 (V1–V28, Time, Amount)</td></tr>
        </table></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="transaction-card">
        <div style="font-size:16px;font-weight:700;color:#1a1f3a;margin-bottom:16px;">Methods compared</div>
        <div style="font-size:13px;font-weight:600;color:#6b7280;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Machine learning models</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
            <span style="background:#eff6ff;color:#1d4ed8;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">Logistic Regression</span>
            <span style="background:#eff6ff;color:#1d4ed8;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">Random Forest</span>
            <span style="background:#eff6ff;color:#1d4ed8;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">XGBoost</span>
        </div>
        <div style="font-size:13px;font-weight:600;color:#6b7280;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Imbalance techniques</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px;">
            <span style="background:#f0fdf4;color:#16a34a;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">Random Undersampling</span>
            <span style="background:#f0fdf4;color:#16a34a;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">SMOTE</span>
            <span style="background:#f0fdf4;color:#16a34a;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">Cost-Sensitive</span>
        </div>
        <div style="font-size:13px;font-weight:600;color:#6b7280;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Explainability</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:#faf5ff;color:#7c3aed;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">SHAP Global</span>
            <span style="background:#faf5ff;color:#7c3aed;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">SHAP Local</span>
            <span style="background:#faf5ff;color:#7c3aed;padding:6px 14px;border-radius:20px;font-size:13px;font-weight:500;">Cross-condition</span>
        </div></div>""", unsafe_allow_html=True)
        st.markdown("""<div class="transaction-card" style="margin-top:16px;">
        <div style="font-size:16px;font-weight:700;color:#1a1f3a;margin-bottom:16px;">Best results (tuned model)</div>
        <table style="width:100%;font-size:14px;">
        <tr><td style="color:#6b7280;padding:6px 0;">Model</td><td style="font-weight:600;color:#dc2626;padding:6px 0;">XGBoost + Cost-Sensitive</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Precision</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">0.908</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">Recall</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">0.827</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">F1 Score</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">0.865</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">AUROC</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">0.982</td></tr>
        <tr><td style="color:#6b7280;padding:6px 0;">AUPRC</td><td style="font-weight:600;color:#1a1f3a;padding:6px 0;">0.885</td></tr>
        </table></div>""", unsafe_allow_html=True)
