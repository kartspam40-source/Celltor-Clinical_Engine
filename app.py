import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SYSTEM CONFIGURATION
st.set_page_config(page_title="CellTor Clinical Portal", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0b0e14; color: #e6edf3; }
    div[data-testid="metric-container"] { background-color: #161b22; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    div[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: bold; }
    h1, h2, h3 { color: #f0883e !important; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD ENGINE
@st.cache_resource
def load_celltor_engine():
    model_path = 'celltor_fair_model_v1.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_celltor_engine()

# 3. SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2900/2900018.png", width=70)
    st.title("CellTor")
    st.caption("Batch Audit & Inference Engine")
    st.markdown("---")
    st.info("**System Status:** [Operational]")

st.title("🧬 CellTor: Genomic Bias Mitigation Engine")

if model is None:
    st.error("### ⚠️ Critical Error: Model Artifact Not Found")
    st.stop()

# 4. TABS SETUP
tab_audit, tab_sandbox, tab_tech = st.tabs(["📁 Dataset Audit & Search", "🎚️ Manual Sandbox", "🧪 Methodology"])

# --- TAB 1: DATASET AUDIT & RSID SEARCH ---
with tab_audit:
    st.markdown("### 1. Upload Genomic Dataset")
    st.write("Upload a CSV containing `RSID`, `AFR_Frequency`, and `EUR_Frequency` to initiate the audit.")
    
    uploaded_file = st.file_uploader("Upload ClinVar/1000G CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if not {'RSID', 'AFR_Frequency', 'EUR_Frequency'}.issubset(df.columns):
            st.error("CSV must contain columns: 'RSID', 'AFR_Frequency', 'EUR_Frequency'")
        else:
            st.success(f"Dataset Loaded: {len(df)} variants processed.")
            
            # Global Bias Detection
            st.markdown("### 2. Global Bias Audit")
            avg_afr = df['AFR_Frequency'].mean()
            avg_eur = df['EUR_Frequency'].mean()
            global_disparity = avg_eur / avg_afr if avg_afr > 0 else 1.0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg AFR Frequency", f"{avg_afr:.4f}")
            c2.metric("Avg EUR Frequency", f"{avg_eur:.4f}")
            c3.metric("Dataset Disparity Ratio", f"{global_disparity:.1f}x")
            
            if global_disparity > 2.0:
                st.warning(f"⚠️ **Systemic Bias Detected:** European signal is {global_disparity:.1f}x stronger. CellTor weights actively applying.")
            else:
                st.success("✅ **Balanced Dataset:** No systemic ancestral skew detected.")
            
            st.divider()
            
            # RSID Search
            st.markdown("### 3. Clinical Inference (Search)")
            search_rsid = st.text_input("🔍 Search Variant by RSID (e.g., rs123)")
            
            if search_rsid:
                variant_data = df[df['RSID'] == search_rsid]
                
                if not variant_data.empty:
                    afr_val = variant_data.iloc[0]['AFR_Frequency']
                    eur_val = variant_data.iloc[0]['EUR_Frequency']
                    
                    st.write(f"**Data found for {search_rsid}:** AFR = {afr_val:.4f} | EUR = {eur_val:.4f}")
                    
                    input_data = pd.DataFrame([[afr_val, eur_val]], columns=['AFR_Frequency', 'EUR_Frequency'])
                    prob = model.predict_proba(input_data)[0][1]
                    prediction = "PATHOGENIC" if prob > 0.5 else "BENIGN"
                    
                    st.markdown("#### 🩺 Diagnosis")
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Clinical Status", prediction)
                    sc2.metric("CellTor Confidence", f"{prob:.2%}")
                else:
                    st.error("RSID not found in the uploaded dataset.")

# --- TAB 2: MANUAL SANDBOX (UPGRADED WITH INSIGHTS & VIZ) ---
with tab_sandbox:
    st.markdown("### Manual Variant Override")
    st.write("Simulate hypothetical frequencies to observe the fairness engine in action.")
    
    col_sliders, col_results = st.columns([1, 2])
    
    with col_sliders:
        man_afr = st.slider("African Allele Freq (AFR_AF)", 0.0, 1.0, 0.0125, format="%.4f", key="s_afr")
        man_eur = st.slider("European Allele Freq (EUR_AF)", 0.0, 1.0, 0.0450, format="%.4f", key="s_eur")
        run_sim = st.button("🚀 Run Simulation", use_container_width=True)

    with col_results:
        if run_sim:
            # 1. Math & Inference
            man_input = pd.DataFrame([[man_afr, man_eur]], columns=['AFR_Frequency', 'EUR_Frequency'])
            man_prob = model.predict_proba(man_input)[0][1]
            man_pred = "PATHOGENIC" if man_prob > 0.5 else "BENIGN"
            man_disparity = man_eur / man_afr if man_afr > 0 else 10.0
            
            # 2. Top Level Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Status", man_pred)
            m2.metric("Confidence", f"{man_prob:.2%}")
            m3.metric("Disparity", f"{man_disparity:.1f}x")
            
            # 3. Dynamic Clinical Insights
            st.markdown("#### 🧠 Clinical Insights")
            if man_eur > (man_afr * 1.5):
                st.warning(f"**Historical Data Skew Detected:** The European signal is **{man_disparity:.1f}x** stronger. In traditional AI models trained on majority-European data, the African genetic context would be treated as background noise. CellTor's Inverse-Probability Weighting actively compensates for this disparity to prevent a 'False-Benign' misdiagnosis.")
            elif man_afr > (man_eur * 1.5):
                rev_disp = man_afr / man_eur if man_eur > 0 else 10.0
                st.info(f"**Minority Variant Highlighted:** The African signal is **{rev_disp:.1f}x** stronger. Because this variant is relatively rare in major European databases, standard models often lack the training context to flag it accurately. CellTor ensures this biological signal is given equal diagnostic weight.")
            else:
                st.success("**Equitable Distribution:** The frequencies between populations are balanced. No significant ancestral bias detected. Standard predictive weights are applied to determine pathogenicity.")
            
            # 4. The Bar Chart
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(x=['AFR Frequency', 'EUR Frequency'], y=[man_afr, man_eur], palette=['#f85149', '#58a6ff'], ax=ax)
            ax.set_facecolor('#0b0e14'); fig.patch.set_facecolor('#0b0e14')
            plt.title("Ancestral Distribution Simulation", color='#e6edf3', fontsize=12)
            ax.tick_params(colors='#e6edf3')
            st.pyplot(fig)

# --- TAB 3: METHODOLOGY ---
with tab_tech:
    st.header("Technical Framework")
    st.markdown("### Inverse-Probability Sample Reweighting")
    st.write("Variants are assigned a fairness weight during training to equalize ancestral influence.")
    st.latex(r"W_i = \begin{cases} \frac{\mu_{EUR}}{\mu_{AFR}} & \text{if } AFR\_AF < EUR\_AF \\ 1.0 & \text{otherwise} \end{cases}")