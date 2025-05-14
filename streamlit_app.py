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
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# --- Secrets & config ---
deepseek_api_key = st.secrets['DEEPSEEK_API_KEY']
model_name       = st.secrets['MODEL_NAME']
openai_api_base  = st.secrets['OPENAI_API_BASE']

st.set_page_config(page_title="AFib Risk Prediction", layout="wide")

# --- Session‐state defaults ---
if 'form_submitted' not in st.session_state:
    st.session_state['form_submitted'] = False
if 'form_values' not in st.session_state:
    st.session_state['form_values'] = {}
if 'user_query_submitted' not in st.session_state:
    st.session_state['user_query_submitted'] = False

# --- Load model & data ---
model = joblib.load("model.pkl")
data  = pd.read_csv("synthetic_data.csv")

# --- Helper functions (PCA, plotting, prediction, context gen) ---
def create_pca_for_plotting(df, input_keys):    
    # copy & select only the keys that actually exist
    d = df.copy()
    common_cols = [c for c in d.columns if c in input_keys]
    # fit a scaler to those columns
    plot_scaler = StandardScaler()
    x_scaled = plot_scaler.fit_transform(d[common_cols])
    # fit PCA and grab the first two components
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)
    d["PC1"] = x_pca[:, 0]
    d["PC2"] = x_pca[:, 1]
    return d, plot_scaler, pca, common_cols

def transform_new_input_for_plotting(new_input, common_cols, plot_scaler, pca):
    d_new = pd.DataFrame([new_input])
    x_scaled = plot_scaler.transform(d_new[common_cols])
    x_pca = pca.transform(x_scaled)
    return x_pca[0, 0], x_pca[0, 1]

def plot_pca_with_af_colors(df_plot, x_new, y_new):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_yes = df_plot[df_plot["outcome_afib_aflutter_new_post"] == 1]
    df_no  = df_plot[df_plot["outcome_afib_aflutter_new_post"] == 0]
    # plot existing patients
    ax.scatter(df_yes["PC1"], df_yes["PC2"], c="#db6459", alpha=0.25, label="AF: Yes")
    ax.scatter(df_no["PC1"], df_no["PC2"], c="#989898", alpha=0.25, label="AF: No")
    # highlight the new patient on top
    ax.scatter(x_new, y_new, c="#db6459", marker=".", s=750, label="New Patient")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Plot, Highlighting New Patient")
    ax.legend()
    st.pyplot(fig)

def make_prediction(form_values):
    input_data = pd.DataFrame([form_values])
    if hasattr(model, "predict_proba"):
        risk_score = model.predict_proba(input_data)[:, 1][0]
    else:
        risk_score = model.predict(input_data)[0]
    if risk_score <= 0.33:
        prediction = "🟢 Low Risk"
    elif risk_score <= 0.66:
        prediction = "🟡 Medium Risk"
    else:
        prediction = "🔴 High Risk"
    estimated_life_years = (1 - risk_score) * 10
    return prediction, risk_score, estimated_life_years

def display_results(pid, prediction, risk_score, estimated_life_years):
    st.subheader(f"Prediction Summary for Patient `{pid}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("AFib Risk Level:", prediction)
    c2.metric("Risk Probability:", f"{risk_score:.2f}")
    c3.metric("Expected AFib-Free Years:", f"{estimated_life_years:.1f} yrs")

def plot_distribution_with_afib_hue(df, form_values, feature_name, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    custom_palette = {0: "#989898", 1: "#db6459"}
    sns.histplot(
        data=df,
        x=feature_name,
        hue="outcome_afib_aflutter_new_post",
        palette=custom_palette,
        bins=50,
        ax=ax,
        kde=False,
        multiple="stack",
        alpha=0.6
    )
    ax.axvline(
        form_values[feature_name],
        color="#db6459",
        linestyle="--",
        linewidth=2,
        label=f"Patient Value: {form_values[feature_name]}"
    )
    afib_absent = mpatches.Patch(color=custom_palette[0], label="AFib Absent")
    afib_present = mpatches.Patch(color=custom_palette[1], label="AFib Present")
    patient_line = mlines.Line2D([], [], color='#db6459', linestyle='--', linewidth=2, label=f"Patient Value: {form_values[feature_name]}")
    ax.legend(handles=[afib_absent, afib_present, patient_line])
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

def generate_patient_context(values):
    context = f"""
    The patient {values['patient_id']} is {values['demographics_age_index_ecg']} and sex {values['demographics_birth_sex']}.
    The patient has a heart rate of {values['ecg_resting_hr']}, PR interval value of {values['ecg_resting_pr']}, QRS curve value of {values['ecg_resting_qrs']}, and corrected QT value of {values['ecg_resting_qtc']}
    """
    conditions = {
        "myocarditis_icd10_prior": "A prior history of Acute Myocarditis",
        "pericarditis_icd10_prior": "A prior history of Acute Pericarditis",
        "aortic_dissection_icd10_prior": "A prior history of Aortic Dissection",
        "ecg_resting_paced": "Heart rhythm is paced",
        "ecg_resting_bigeminy": "Heart rhythm is bigeminy",
        "ecg_resting_LBBB": "LBBB QRS Morphology",
        "ecg_resting_RBBB": "RBBB QRS Morphology",
        "ecg_resting_incomplete_LBBB": "Incomplete LBBB QRS Morphology",
        "ecg_resting_incomplete_RBBB": "Incomplete RBBB QRS Morphology",
        "ecg_resting_LAFB": "LAFB QRS Morphology",
        "ecg_resting_LPFB": "LPFB QRS Morphology",
        "ecg_resting_bifascicular_block": "Bifascicular block in heart rhythm",
        "ecg_resting_trifascicular_block": "Trifascular block in heart rhythm",
        "ecg_resting_intraventricular_conduction_delay": "Intraventricular condition delay in heart rhythm",
        "event_cv_hf_admission_icd10_prior": "A history of heart failure admission",
        "event_cv_cad_acs_acute_mi_icd10_prior": "History of myocardial infarction",
        "event_cv_cad_acs_unstable_angina_icd10_prior": "History of unstable angine",
        "event_cv_cad_acs_other_icd10_prior": "history of other acute coronary syndrome",
        "event_cv_ep_vt_any_icd10_prior": "History of ventriculat tachycardia",
        "event_cv_ep_sca_survived_icd10_cci_prior": "Has survived sudden cardiac arrest",
        "event_cv_cns_stroke_ischemic_icd10_prior": "History of Acute ischemic stroke",
        "event_cv_cns_stroke_hemorrh_icd10_prior": "History of Acute hemorrhagic stroke",
        "event_cv_cns_tia_icd10_prior": "Has had a transient ischemic attack",
        "pci_prior": "Has had percutaneous coronary intervention",
        "cabg_prior": "Has had coronary artery bypass grafting procedure",
        "transplant_heart_cci_prior": "Has had a heart transplant",
        "lvad_cci_prior": "Has had LVAD implantation procedure",
        "pacemaker_permanent_cci_prior": "Have a permanent pacemaker implantation",
        "crt_cci_prior": "Has had a cardiac resynchronization therapy (CRT) Implantation",
        "icd_cci_prior": "Has had Internal Cardioverter defibrillator implantation",
    }
 
    condition_statements = [
        f"The patient has {desc}." for key, desc in conditions.items() if values.get(key, 0) == 1
    ]
 
    if condition_statements:
        context += "\n" + "\n".join(condition_statements)
    else:
        context += "\nThe patient has no conditions on record"
   
    return context.strip()

default_values = {
    "patient_id": None,
    "demographics_age_index_ecg": None,
    "demographics_birth_sex": None,
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

form_values = default_values.copy()

def is_valid(value):
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    return True

mandatory_fields = [
    "patient_id",
    "demographics_age_index_ecg",
    "ecg_resting_hr",
    "ecg_resting_pr",
    "ecg_resting_qrs",
    "ecg_resting_qtc"
]

img_c1, img_c2, img_c3 = st.columns(3)
with img_c2:
    st.image("title.png", width=300)

st.title("Risk Prediction for Atrial Fibrillation")
st.badge("All fields marked with ⚠️ are required. Please fill them out before submitting.", color="gray")

if "form_key" not in st.session_state:
    st.session_state["form_key"] = str(uuid.uuid4())

def unique_key(base):
    return f"{base}_{st.session_state['form_key']}"

def render_form():
    with st.form(key=st.session_state["form_key"], clear_on_submit=False):
        st.subheader("Patient ID & Demographics")
        pi_c1, pi_c2, pi_c3 = st.columns(3)
        form_values["patient_id"] = pi_c1.text_input("Patient ID ⚠️", key=unique_key("patient_id"))
        g_options = {"Male": 0, "Female": 1}
        sel_gender = pi_c2.selectbox("Gender ⚠️", list(g_options.keys()))
        form_values["demographics_birth_sex"] = g_options[sel_gender]
        form_values["demographics_age_index_ecg"] = pi_c3.number_input("Age (years) ⚠️", min_value=0, max_value=120, value=0)
        st.divider()
 
        st.subheader("Cardiovascular Diseases")
        st.caption("Select all that apply.")
        cd_c1, cd_c2, cd_c3 = st.columns(3)
        form_values["myocarditis_icd10_prior"] = 1 if cd_c1.checkbox("Acute myocarditis") else 0
        form_values["pericarditis_icd10_prior"] = 1 if cd_c2.checkbox("Acute pericarditis") else 0
        form_values["aortic_dissection_icd10_prior"] = 1 if cd_c3.checkbox("Aortic dissection") else 0
        st.divider()
 
        st.subheader("Cardiovascular Events & Procedures")
        st.caption("Select all that apply.")
        c1_ced, c2_ced, c3_ced = st.columns(3)
        form_values["event_cv_hf_admission_icd10_prior"] = 1 if c1_ced.checkbox("Heart failure admission") else 0
        form_values["event_cv_cad_acs_acute_mi_icd10_prior"] = 1 if c1_ced.checkbox("Acute myocardial infarction") else 0
        form_values["event_cv_cad_acs_unstable_angina_icd10_prior"] = 1 if c1_ced.checkbox("Unstable angina") else 0
        form_values["event_cv_cad_acs_other_icd10_prior"] = 1 if c1_ced.checkbox("Other acute coronary syndrome") else 0
        form_values["event_cv_ep_vt_any_icd10_prior"] = 1 if c1_ced.checkbox("Ventricular tachycardia") else 0
        form_values["event_cv_ep_sca_survived_icd10_cci_prior"] = 1 if c2_ced.checkbox("Survived sudden cardiac arrest") else 0
        form_values["event_cv_cns_stroke_ischemic_icd10_prior"] = 1 if c2_ced.checkbox("Acute ischemic stroke") else 0
        form_values["event_cv_cns_stroke_hemorrh_icd10_prior"] = 1 if c2_ced.checkbox("Acute hemorrhagic stroke") else 0
        form_values["event_cv_cns_tia_icd10_prior"] = 1 if c2_ced.checkbox("Transient ischemic attack (TIA)") else 0
        form_values["pci_prior"] = 1 if c3_ced.checkbox("Percutaneous coronary intervention (PCI)") else 0
        form_values["cabg_prior"] = 1 if c3_ced.checkbox("Coronary artery bypass grafting (CABG)") else 0
        form_values["transplant_heart_cci_prior"] = 1 if c3_ced.checkbox("Heart transplantation") else 0
        form_values["lvad_cci_prior"] = 1 if c3_ced.checkbox("LVAD implantation") else 0
        st.divider()
 
        st.subheader("Cardiovascular Devices")
        st.caption("Select all that apply.")
        cdev_c1, cdev_c2, cdev_c3 = st.columns(3)
        form_values["pacemaker_permanent_cci_prior"] = 1 if cdev_c1.checkbox("Permanent pacemaker") else 0
        form_values["crt_cci_prior"] = 1 if cdev_c2.checkbox("CRT device") else 0
        form_values["icd_cci_prior"] = 1 if cdev_c3.checkbox("Implantable cardioverter‑defibrillator (ICD)") else 0
        st.divider()
 
        st.subheader("12‑Lead ECG Measurements")
        ecg_c1, ecg_c2 = st.columns(2)
        form_values["ecg_resting_hr"] = ecg_c1.number_input("Heart Rate (bpm) ⚠️", step=1, value=None)
        form_values["ecg_resting_pr"] = ecg_c1.number_input("PR Interval (ms) ⚠️", min_value=0, value=None)
        form_values["ecg_resting_qrs"] = ecg_c2.number_input("QRS Duration (ms) ⚠️", min_value=0, value=None)
        form_values["ecg_resting_qtc"] = ecg_c2.number_input("QTc Interval (ms) ⚠️", min_value=0, value=None)
        st.divider()
 
        st.subheader("ECG Morphology & Conduction")
        st.caption("Select all that apply.")
        ecg_c3, ecg_c4, ecg_c5 = st.columns(3)
        form_values["ecg_resting_paced"] = 1 if ecg_c3.checkbox("Paced rhythm") else 0
        form_values["ecg_resting_bigeminy"] = 1 if ecg_c3.checkbox("Bigeminy") else 0
        form_values["ecg_resting_LBBB"] = 1 if ecg_c3.checkbox("LBBB") else 0
        form_values["ecg_resting_RBBB"] = 1 if ecg_c3.checkbox("RBBB") else 0
        form_values["ecg_resting_incomplete_LBBB"] = 1 if ecg_c4.checkbox("Incomplete LBBB") else 0
        form_values["ecg_resting_incomplete_RBBB"] = 1 if ecg_c4.checkbox("Incomplete RBBB") else 0
        form_values["ecg_resting_LAFB"] = 1 if ecg_c4.checkbox("LAFB") else 0
        form_values["ecg_resting_LPFB"] = 1 if ecg_c4.checkbox("LPFB") else 0
        form_values["ecg_resting_bifascicular_block"] = 1 if ecg_c5.checkbox("Bifascicular block") else 0
        form_values["ecg_resting_trifascicular_block"] = 1 if ecg_c5.checkbox("Trifascicular block") else 0
        form_values["ecg_resting_intraventricular_conduction_delay"] = 1 if ecg_c5.checkbox("Intraventricular conduction delay") else 0
 
        submit_flag = st.form_submit_button("Submit for Risk Prediction 🩺")
        save_flag = st.form_submit_button("Save Patient Record ☁️")
        return submit_flag, save_flag

form_container = st.empty()

with form_container:
    submit_flag, save_flag = render_form()

if submit_flag:
    if any(not is_valid(form_values[f]) for f in mandatory_fields):
        st.error("Please complete all mandatory fields with valid values.")
    else:
        st.session_state["form_submitted"] = True
        st.session_state["form_values"] = form_values.copy()

if st.session_state.get("form_submitted", False):
    try:
        tab1, tab2 = st.tabs(["Summary","Read More"])
 
        # — SUMMARY TAB —
        with tab1:
            vals = st.session_state["form_values"]
            lvl, score, yrs = make_prediction(vals)
            display_results(vals["patient_id"], lvl, score, yrs)
 
            # PCA + Age
            df_plot, scaler, pca, cols = create_pca_for_plotting(data, vals.keys())
            x_new, y_new = transform_new_input_for_plotting(vals, cols, scaler, pca)
            c1,c2 = st.columns(2)
            with c1:
                plot_pca_with_af_colors(df_plot, x_new, y_new)
            with c2:
                plot_distribution_with_afib_hue(data, vals, "demographics_age_index_ecg", "Age (y)")
 
            # ECG distributions
            c1,c2 = st.columns(2)
            with c1:
                plot_distribution_with_afib_hue(data, vals, "ecg_resting_hr", "Heart Rate (bpm)")
                plot_distribution_with_afib_hue(data, vals, "ecg_resting_qrs", "QRS Duration (ms)")
            with c2:
                plot_distribution_with_afib_hue(data, vals, "ecg_resting_pr", "PR Interval (ms)")
                plot_distribution_with_afib_hue(data, vals, "ecg_resting_qtc", "QTc Interval (ms)")
            st.badge("⚠️ All distributions and PCA backdrops are simulated and do not represent the actual training or evaluation data. They were created to mimic real-world patterns while ensuring data privacy.",
                     color="gray")
 
            # Q&A UI
            question = st.text_area("Ask about your health report")
            if st.button("Ask Question", key="ask_question_btn"):
                st.session_state["user_query_submitted"] = True
 
            if st.session_state["user_query_submitted"] and question.strip():
                vals = st.session_state["form_values"]
                context = generate_patient_context(vals)
                template = """
                You are a helpful medical assistant.
                Use the patient's data to answer their questions clearly.
                Search for answers in the internet if needed to answer the questions
                Patient Data: {context}
                Question: {question}
                Answer:
                """
 
                prompt = PromptTemplate(input_variables=["context", "question"], template=template)
 
                llm = ChatOpenAI(
                    openai_api_base=openai_api_base,
                    openai_api_key=deepseek_api_key,
                    model=model_name,
                    temperature=0.7
                )
 
                chain = prompt | llm
               
                with st.spinner("Generating response…"):
                    response = chain.invoke({"context": context, "question": question})
                    st.markdown(response.content)
 
        # — READ MORE TAB —
        with tab2:
            with st.expander("Risk Model & Feature Set"):
                st.markdown("""
                    An **XGBoost** classifier was trained on the following groups of features to estimate your risk of developing atrial fibrillation (AFib):
                    - **Demographics**: Age at time of ECG, biological sex  
                    - **Disease History**: Acute myocarditis, pericarditis, aortic dissection  
                    - **Cardiovascular Events & Procedures**: Heart failure admission, acute myocardial infarction (MI), unstable angina, stroke, transient ischemic attack (TIA), PCI, CABG, LVAD implantation, heart transplantation  
                    - **Implanted Devices**: Permanent pacemaker, CRT device, implantable cardioverter-defibrillator (ICD)  
                    - **ECG Measurements**: Heart rate (bpm), PR interval (ms), QRS duration (ms), QTc interval (ms)  
                    - **ECG Conduction Abnormalities**: Paced rhythm, bigeminy, LBBB, RBBB, incomplete blocks, LAFB, LPFB, bifascicular/trifascicular block, intraventricular conduction delay  
                    The model outputs a probability of new-onset AFib, categorized as **Low** 🟢, **Medium** 🟡, or **High** 🔴 risk.
                    """
                )
            with st.expander("How to Read the Visualizations"):
                st.markdown(
                    """
                    **PCA Projection**  
                    - Reduces all numeric inputs into two principal components (PC1 & PC2).  
                    - Background dots = synthetic patient cohort (AFib vs. no AFib).  
                    - Large red dot = your individual feature profile.
        
                    **Outcome‑Stratified Histograms**  
                    For each of your ECG values, we show where you fall relative to the simulated AFib and non‑AFib populations:  
                    - Age (years)  
                    - Heart Rate (bpm)  
                    - PR Interval (ms)  
                    - QRS Duration (ms)  
                    - QTc Interval (ms)  
                    """
                )
            with st.expander("ECG Feature Definitions"):
                st.markdown(
                    """
                    - **Heart Rate (bpm)**: Beats per minute  
                    - **PR Interval (ms)**: Time from atrial to ventricular depolarization  
                    - **QRS Duration (ms)**: Time for ventricular depolarization  
                    - **QTc Interval (ms)**: QT interval corrected for heart rate  
                    """
                )
            with st.expander("Synthetic Data Disclaimer"):
                st.markdown(
                    """
                    All cohort distributions and PCA backdrops are **synthetic** and do **not** reflect any real patient records.  
                    They were generated solely to illustrate your profile's position within a plausible population, while preserving privacy.
                    """
                )
            with st.expander("Chatbot Overview"):
                st.markdown("""
                This AI-powered chatbot is designed to **interpret structured clinical and ECG data** and generate **clear, concise medical summaries**. It provides a natural-language explanation of key health indicators, making it suitable for both **clinical support** and **non-clinical understanding**.
 
                **What It Does**
                           
                When provided with inputs such as age, sex, heart rate, PR interval, QRS duration, QTc, and medical history, the chatbot returns a **human-readable summary** that includes:
 
                - **Vital Signs Interpretation** explains whether values like heart rate, PR interval, QRS duration, and QTc are within normal ranges.
 
                - **Abnormality Detection** highlights clinical concerns such as bradycardia, heart block, or QT prolongation, if present.
 
                - **Clinical Recommendations** offers actionable insights such as the need for urgent evaluation, pacemaker consideration, or further diagnostic testing.
 
                - **Contextual Clarification** includes assumptions made in interpreting coded values (e.g., sex = "0" interpreted as female).
 
                **Use Cases**
                1. **Clinical Decision Support** assists healthcare providers in quickly understanding ECG findings and identifying red flags in patient data.
 
                2. **Medical Education & Training** acts as a learning tool for students and trainees by explaining ECG parameters in plain language.
 
                3. **Remote Monitoring Summaries** converts raw telemetry data from wearable devices or remote patient monitors into digestible reports.
 
                4. **Triage Assistance** helps non-specialist medical staff or first responders make informed decisions by flagging urgent conditions.
 
                5. **Second Opinion for Practitioners** provides automated, unbiased review of ECG data to complement human judgment.
 
                6. **Patient-Facing Summaries (with Caution)** can be used to generate simplified explanations of ECG results for patients, though should be reviewed by a clinician.
                """)
 
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- Save placeholder ---
if save_flag:
    st.info("Saving is disabled.")

# --- Clear Form handler must also reset flags ---
if st.button("Clear Form for New Entry 🗑️", key="clear_form_btn"):
    st.session_state["form_key"]             = str(uuid.uuid4())
    st.session_state["form_submitted"]       = False
    st.session_state["user_query_submitted"] = False
    form_container.empty()
    with form_container:
        submit_flag, save_flag = render_form()