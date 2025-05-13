import os
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from custom_transformers import PreprocessDataTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- secrets & config ---
deepseek_api_key = st.secrets['DEEPSEEK_API_KEY']
model_name       = st.secrets['MODEL_NAME']
openai_api_base  = st.secrets['OPENAI_API_BASE']
st.set_page_config(page_title="AFib Risk Prediction", layout="wide")

# --- session state defaults ---
if 'form_submitted' not in st.session_state:
    st.session_state['form_submitted'] = False
if 'form_values' not in st.session_state:
    st.session_state['form_values'] = {}
if 'question' not in st.session_state:
    st.session_state['question'] = ""
if 'user_query_submitted' not in st.session_state:
    st.session_state['user_query_submitted'] = False

# --- load model & data ---
model = joblib.load("model.pkl")
data  = pd.read_csv("synthetic_data.csv")

# --- helper functions ---
def create_pca_for_plotting(df, input_keys):    
    d = df.copy()
    common_cols = [c for c in d.columns if c in input_keys]
    scaler = StandardScaler().fit(d[common_cols])
    pcs = PCA(n_components=2).fit(scaler.transform(d[common_cols]))
    coords = pcs.transform(scaler.transform(d[common_cols]))
    d["PC1"], d["PC2"] = coords[:,0], coords[:,1]
    return d, scaler, pcs, common_cols

def transform_new_input_for_plotting(new_input, common_cols, scaler, pcs):
    arr = scaler.transform(pd.DataFrame([new_input])[common_cols])
    pc1,pc2 = pcs.transform(arr)[0]
    return pc1, pc2

def plot_pca_with_af_colors(df_plot, x_new, y_new):
    fig, ax = plt.subplots(figsize=(8,5))
    yes = df_plot[df_plot["outcome_afib_aflutter_new_post"]==1]
    no  = df_plot[df_plot["outcome_afib_aflutter_new_post"]==0]
    ax.scatter(yes["PC1"], yes["PC2"], c="#db6459", alpha=0.25, label="AF: Yes")
    ax.scatter(no["PC1"],  no["PC2"],  c="#989898", alpha=0.25, label="AF: No")
    ax.scatter(x_new, y_new, c="#db6459", s=200, label="You")
    ax.set(title="PCA Plot", xlabel="PC1", ylabel="PC2")
    ax.legend()
    st.pyplot(fig)

def plot_distribution_with_afib_hue(df, fv, feat, title):
    fig, ax = plt.subplots(figsize=(8,5))
    pal = {0:"#989898",1:"#db6459"}
    sns.histplot(df, x=feat, hue="outcome_afib_aflutter_new_post",
                 palette=pal, bins=30, ax=ax, alpha=0.6, multiple="stack")
    ax.axvline(fv[feat], color="#db6459", linestyle="--", linewidth=2,
               label=f"You: {fv[feat]}")
    ax.set(title=title, xlabel=feat, ylabel="Count")
    ax.legend()
    st.pyplot(fig)

def make_prediction(fv):
    df = pd.DataFrame([fv])
    if hasattr(model,"predict_proba"):
        p = model.predict_proba(df)[:,1][0]
    else:
        p = model.predict(df)[0]
    label = "🟢 Low" if p<=0.33 else "🟡 Med" if p<=0.66 else "🔴 High"
    yrs = (1-p)*10
    return label, p, yrs

def generate_patient_context(v):
    return (
        f"Patient {v['patient_id']}, age {v['demographics_age_index_ecg']}, "
        f"sex {v['demographics_birth_sex']}. "
        f"ECG HR {v['ecg_resting_hr']}, PR {v['ecg_resting_pr']}, "
        f"QRS {v['ecg_resting_qrs']}, QTc {v['ecg_resting_qtc']}."
    )

# --- defaults & validation ---
default_values = {
    "patient_id": None,
    "demographics_birth_sex": None,
    "demographics_age_index_ecg": None,
    "myocarditis_icd10_prior": 0,
    "pericarditis_icd10_prior": 0,
    "aortic_dissection_icd10_prior": 0,
    "event_cv_hf_admission_icd10_prior": 0,
    "event_cv_cad_acs_acute_mi_icd10_prior": 0,
    "event_cv_cad_acs_unstable_angina_icd10_prior": 0,
    "event_cv_cad_acs_other_icd10_prior": 0,
    "event_cv_ep_vt_any_icd10_prior": 0,
    "event_cv_ep_sca_survived_icd10_cci_prior": 0,
    "event_cv_cns_stroke_ischemic_icd10_prior": 0,
    "event_cv_cns_stroke_hemorrh_icd10_prior": 0,
    "event_cv_cns_tia_icd10_prior": 0,
    "pci_prior": 0,
    "cabg_prior": 0,
    "transplant_heart_cci_prior": 0,
    "lvad_cci_prior": 0,
    "pacemaker_permanent_cci_prior": 0,
    "crt_cci_prior": 0,
    "icd_cci_prior": 0,
    "ecg_resting_hr": None,
    "ecg_resting_pr": None,
    "ecg_resting_qrs": None,
    "ecg_resting_qtc": None,
    "ecg_resting_paced": 0,
    "ecg_resting_bigeminy": 0,
    "ecg_resting_LBBB": 0,
    "ecg_resting_RBBB": 0,
    "ecg_resting_incomplete_LBBB": 0,
    "ecg_resting_incomplete_RBBB": 0,
    "ecg_resting_LAFB": 0,
    "ecg_resting_LPFB": 0,
    "ecg_resting_bifascicular_block": 0,
    "ecg_resting_trifascicular_block": 0,
    "ecg_resting_intraventricular_conduction_delay": 0
}
def is_valid(v):
    return v is not None and not (isinstance(v,str) and v.strip()=="")
mandatory = [
    "patient_id","demographics_age_index_ecg",
    "ecg_resting_hr","ecg_resting_pr",
    "ecg_resting_qrs","ecg_resting_qtc"
]

# --- UI ---
st.title("Risk Prediction for Atrial Fibrillation")
st.image("title.png", width=300)

# 1) Input form
if not st.session_state['form_submitted']:
    form_key = str(uuid.uuid4())
    with st.form(key=form_key):
        st.subheader("Enter Patient Data ⚠️")
        fv = default_values.copy()

        fv["patient_id"]               = st.text_input("Patient ID ⚠️")
        sex_sel                        = st.selectbox("Sex ⚠️", ["Male","Female"])
        fv["demographics_birth_sex"]   = 0 if sex_sel=="Male" else 1
        fv["demographics_age_index_ecg"]= st.number_input("Age (years) ⚠️", min_value=0, max_value=120, value=0)

        st.divider(); st.subheader("Cardiovascular Diseases")
        c1,c2,c3 = st.columns(3)
        fv["myocarditis_icd10_prior"]         = 1 if c1.checkbox("Myocarditis") else 0
        fv["pericarditis_icd10_prior"]        = 1 if c2.checkbox("Pericarditis") else 0
        fv["aortic_dissection_icd10_prior"]   = 1 if c3.checkbox("Aortic Dissection") else 0

        st.divider(); st.subheader("Events & Procedures")
        e1,e2,e3 = st.columns(3)
        fv["event_cv_hf_admission_icd10_prior"]    = 1 if e1.checkbox("HF Admission") else 0
        fv["event_cv_cad_acs_acute_mi_icd10_prior"]= 1 if e1.checkbox("Acute MI") else 0
        fv["event_cv_cad_acs_unstable_angina_icd10_prior"] = 1 if e1.checkbox("Unstable Angina") else 0
        fv["event_cv_cad_acs_other_icd10_prior"]   = 1 if e1.checkbox("Other ACS") else 0
        fv["event_cv_ep_vt_any_icd10_prior"]       = 1 if e2.checkbox("VT") else 0
        fv["event_cv_ep_sca_survived_icd10_cci_prior"]=1 if e2.checkbox("SCA Survivor") else 0
        fv["event_cv_cns_stroke_ischemic_icd10_prior"]=1 if e2.checkbox("Ischemic Stroke") else 0
        fv["event_cv_cns_stroke_hemorrh_icd10_prior"]=1 if e2.checkbox("Hemorrhagic Stroke") else 0
        fv["event_cv_cns_tia_icd10_prior"]= 1 if e2.checkbox("TIA") else 0
        fv["pci_prior"]               = 1 if e3.checkbox("PCI") else 0
        fv["cabg_prior"]              = 1 if e3.checkbox("CABG") else 0
        fv["transplant_heart_cci_prior"]=1 if e3.checkbox("Transplant") else 0
        fv["lvad_cci_prior"]          = 1 if e3.checkbox("LVAD") else 0

        st.divider(); st.subheader("Devices")
        d1,d2,d3 = st.columns(3)
        fv["pacemaker_permanent_cci_prior"] = 1 if d1.checkbox("Pacemaker") else 0
        fv["crt_cci_prior"]                = 1 if d2.checkbox("CRT") else 0
        fv["icd_cci_prior"]                = 1 if d3.checkbox("ICD") else 0

        st.divider(); st.subheader("ECG Measurements ⚠️")
        ec1,ec2 = st.columns(2)
        fv["ecg_resting_hr"]  = ec1.number_input("Heart Rate (bpm) ⚠️", min_value=0, value=0)
        fv["ecg_resting_pr"]  = ec1.number_input("PR Interval (ms) ⚠️", min_value=0, value=0)
        fv["ecg_resting_qrs"] = ec2.number_input("QRS Duration (ms) ⚠️", min_value=0, value=0)
        fv["ecg_resting_qtc"] = ec2.number_input("QTc Interval (ms) ⚠️", min_value=0, value=0)

        submitted = st.form_submit_button("Submit for Risk 🚀")
        if submitted:
            if any(not is_valid(fv[k]) for k in mandatory):
                st.error("Please complete all required fields.")
            else:
                st.session_state['form_submitted'] = True
                st.session_state['form_values']   = fv

# 2) Summary + Chat
if st.session_state['form_submitted']:
    fv = st.session_state['form_values']
    tab1, tab2 = st.tabs(["Summary","Read More"])
    try:
        with tab1:
            # metrics
            label, prob, yrs = make_prediction(fv)
            c1,c2,c3 = st.columns(3)
            c1.metric("Risk Level", label)
            c2.metric("Risk Probability", f"{prob:.2f}")
            c3.metric("AFib‑free Years", f"{yrs:.1f}")

            # plots
            dfp,sc,pcs,cols = create_pca_for_plotting(data, fv.keys())
            x0,y0 = transform_new_input_for_plotting(fv, cols, sc, pcs)
            plot_pca_with_af_colors(dfp, x0, y0)
            plot_distribution_with_afib_hue(data, fv, "demographics_age_index_ecg", "Age Distribution")
            plot_distribution_with_afib_hue(data, fv, "ecg_resting_hr",           "Heart Rate Distribution")
            plot_distribution_with_afib_hue(data, fv, "ecg_resting_pr",           "PR Interval Distribution")
            plot_distribution_with_afib_hue(data, fv, "ecg_resting_qrs",          "QRS Duration Distribution")
            plot_distribution_with_afib_hue(data, fv, "ecg_resting_qtc",          "QTc Interval Distribution")

            # chat
            st.subheader("Ask about this report")
            st.session_state['question'] = st.text_area(
                "Your question:", 
                value=st.session_state['question'], 
                key="user_question"
            )
            if st.button("Submit Question", key="ask_btn"):
                st.session_state['user_query_submitted'] = True

            if st.session_state['user_query_submitted'] and st.session_state['question'].strip():
                context = generate_patient_context(fv)
                llm = ChatOpenAI(
                    openai_api_base=openai_api_base,
                    openai_api_key=deepseek_api_key,
                    model_name=model_name,
                    temperature=0.7
                )
                prompt = PromptTemplate(
                    input_variables=["context","question"],
                    template="You are a helpful medical assistant.\nContext: {context}\nQuestion: {question}\nAnswer:"
                )
                chain = LLMChain(llm=llm, prompt=prompt)
                with st.spinner("Generating answer..."):
                    ans = chain.run({"context": context, "question": st.session_state['question']})
                st.write(ans)

        with tab2:
            st.header("Learn More About the Dashboard")
            with st.expander("Risk Model & Feature Set"):
                st.markdown("""
- **Demographics**: Age, Biological sex  
- **Disease History**: Myocarditis, Pericarditis, Aortic dissection  
- **Events & Procedures**: HF admission, MI, Angina, Stroke, TIA, PCI, CABG, LVAD, Transplant  
- **Devices**: Pacemaker, CRT, ICD  
- **ECG**: Heart rate, PR, QRS, QTc  
""")
            with st.expander("Visualizations Explained"):
                st.markdown("""
- **PCA**: projects you vs. synthetic cohort  
- **Histograms**: shows your value vs. AF vs. no‑AF  
""")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# 3) Clear button
if st.button("Clear Form for New Entry 🗑️", key="clear_form_btn"):
    for k in ['form_submitted','form_values','question','user_query_submitted']:
        st.session_state.pop(k, None)
    st.experimental_rerun()