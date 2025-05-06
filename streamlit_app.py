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

st.set_page_config(page_title="AFib Risk Prediction", layout="wide")

model = joblib.load("model.pkl")
data = pd.read_csv("synthetic_data.csv")







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
    st.subheader(f"Prediction Summary for Patient: `{pid}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("AFib Risk Level:", prediction)
    c2.metric("Risk Probability:", f"{risk_score:.2f}")
    c3.metric("Expected AFib-Free Years", f"{estimated_life_years:.1f} yrs")

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

        submit_flag = st.form_submit_button("Submit for Risk Prediction 🚀")
        save_flag = st.form_submit_button("Save Patient Record ☁️")
        return submit_flag, save_flag

form_container = st.empty()

with form_container:
    submit_flag, save_flag = render_form()

if submit_flag:
    if any(not is_valid(form_values[f]) for f in mandatory_fields):
        st.error("Please complete all mandatory fields with valid values.")
    else:
        df_input = pd.DataFrame([form_values])
        try:
            tab1, tab2 = st.tabs(["Summary", "Read More"])




            
            with tab1:
                pred, score, life_yrs = make_prediction(form_values)
                display_results(form_values["patient_id"], pred, score, life_yrs)
                st.badge("⚠️ The distributions **shown below** are simulated and **do not reflect the actual data** used to train or evaluate the model. They were generated to approximate real‑world patterns while preserving data privacy.",
                         color="gray")
                c1, c2 = st.columns(2)
                with c1:
                    df_plot, plot_scaler, pca, common_cols = create_pca_for_plotting(data, form_values.keys())
                    x_new, y_new = transform_new_input_for_plotting(form_values, common_cols, plot_scaler, pca)
                    plot_pca_with_af_colors(df_plot, x_new, y_new)
                with c2:
                    plot_distribution_with_afib_hue(data, form_values, "demographics_age_index_ecg", "Age Distribution")

                st.subheader("ECG Feature Distributions Compared to AFib Population")
                c1, c2 = st.columns(2)
                with c1:
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_hr", "Heart Rate (HR) Distribution")
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_qrs", "QRS Duration Distribution")
                with c2:
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_pr", "PR Interval Distribution")
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_qtc", "QTc Interval Distribution")
                st.subheader("ECG Feature Distributions Compared to AFib Population")
                c1, c2, c3, c4 = st.columns(2)
                with c1:
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_hr", "Heart Rate (HR) Distribution")
                with c2:
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_qrs", "QRS Duration Distribution")
                with c3:
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_pr", "PR Interval Distribution")
                with c4:
                    plot_distribution_with_afib_hue(data, form_values, "ecg_resting_qtc", "QTc Interval Distribution")
            with tab2:
                st.write("This section will soon include detailed explanations of the risk models, ECG feature impacts, and interpretation guides.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if save_flag:
    st.info("Saving functionality is currently disabled.")

if st.button("Clear Form for New Entry 🗑️", key="clear_form_btn"):
    st.session_state["form_key"] = str(uuid.uuid4())
    form_container.empty()
    with form_container:
        render_form()
