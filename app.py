




import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Custom CSS  —  Hospital Theme (Times New Roman everywhere)
# ------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Font: Times New Roman (icons untouched) ---------- */
html, body, .stApp, .stMarkdown, p, label, h1, h2, h3, h4, h5, h6, li, td, th,
input, textarea, button, small, b, strong,
div[data-baseweb="select"] *, [data-testid="stWidgetLabel"] *,
[data-testid="stTab"] *, [data-testid="stAlert"] * {
    font-family: 'Times New Roman', Times, serif !important;
}
[data-testid="stExpander"] details summary { display:flex !important; align-items:center !important; gap:10px !important; }
[data-testid="stExpander"] details summary svg { font-family: initial !important; }
[data-testid="stExpander"] details summary p { margin:0 !important; }

/* ---------- Hospital ward background ---------- */
.stApp {
    color:#12333d;
    background:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='46' height='46'%3E%3Cpath d='M20 14h6v6h6v6h-6v6h-6v-6h-6v-6h6z' fill='%230e7c86' fill-opacity='.045'/%3E%3C/svg%3E"),
        linear-gradient(180deg,#f3fafb 0%,#e6f2f5 100%);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1150px; padding-top: 1.6rem; }
h1,h2,h3,h4,h5,h6,p,label { color:#12333d; }

/* ---------- Hospital header ---------- */
.hosp-header {
    display:flex; align-items:center; justify-content:space-between; gap:1.5rem; flex-wrap:wrap;
    background:#ffffff; border:1px solid #cfe3e8; border-top:6px solid #0e7c86;
    border-radius:16px; padding:1.3rem 1.8rem; margin-bottom:1.6rem;
    box-shadow:0 6px 24px rgba(14,124,134,.10);
}
.brand { display:flex; align-items:center; gap:1.1rem; }
.cross { position:relative; width:54px; height:54px; flex:none; border-radius:14px; background:#fdecee; border:1px solid #f4c2c6; }
.cross::before,.cross::after { content:""; position:absolute; background:#d9363e; border-radius:2px; }
.cross::before { left:22px; top:11px; width:10px; height:32px; }
.cross::after  { left:11px; top:22px; width:32px; height:10px; }
.brand small { color:#0e7c86; letter-spacing:.16em; font-size:.78rem; font-weight:700; }
.brand h1 { margin:.15rem 0 .2rem; font-size:2.15rem; color:#0b4f5c; line-height:1.15; }
.brand p  { margin:0; color:#4b6b75; font-size:1.05rem; }
.ecg { width:280px; max-width:100%; height:60px; }
.ecg path { fill:none; stroke-width:2.4; stroke-linecap:round; stroke-linejoin:round; }
.ecg .base { stroke:#cfe3e8; }
.ecg .pulse { stroke:#d9363e; stroke-dasharray:140 260; animation:ecg 3s linear infinite; }
@keyframes ecg { from{stroke-dashoffset:400;} to{stroke-dashoffset:0;} }

/* ---------- Patient admission form (chart card) ---------- */
[data-testid="stForm"] {
    background:#ffffff; border:1px solid #cfe3e8; border-left:7px solid #0e7c86;
    border-radius:16px; padding:1.8rem 2rem; box-shadow:0 6px 24px rgba(14,124,134,.08);
}
[data-testid="stForm"] h3 { color:#0b4f5c; border-bottom:2px dashed #cfe3e8; padding-bottom:.6rem; margin-bottom:1rem; }
label, [data-testid="stWidgetLabel"] p { color:#12333d !important; font-weight:600; font-size:1.05rem; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap:10px; background:transparent; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab"] {
    background:#eef6f8; color:#3d6470; border:1px solid #cfe3e8; border-radius:999px;
    padding:9px 22px; font-size:1.08rem; transition:all .25s ease;
}
.stTabs [data-baseweb="tab"]:hover { background:#dff0f3; }
.stTabs [aria-selected="true"] { background:#0e7c86 !important; color:#ffffff !important; border-color:#0e7c86; box-shadow:0 4px 12px rgba(14,124,134,.3); }
.stTabs [aria-selected="true"] * { color:#ffffff !important; }

/* ---------- Inputs ---------- */
div[data-baseweb="select"] > div { background:#f7fbfc; border:1px solid #b9d3da; border-radius:10px; color:#12333d; }
div[data-baseweb="select"] > div:hover { border-color:#0e7c86; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color:#0e7c86 !important; }

/* ---------- Predict button ---------- */
.stButton>button, [data-testid="stFormSubmitButton"]>button {
    background:linear-gradient(135deg,#0e7c86 0%,#0b5d72 100%); color:#ffffff; border:none;
    padding:.85rem 2rem; font-size:1.3rem; font-weight:700; letter-spacing:.03em;
    border-radius:12px; width:100%; transition:all .25s ease; box-shadow:0 6px 18px rgba(14,124,134,.35);
}
.stButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover {
    transform:translateY(-2px); box-shadow:0 10px 24px rgba(14,124,134,.5); color:#fff;
}

/* ---------- Diagnostic report ---------- */
.report { background:#fff; border:1px solid #cfe3e8; border-radius:16px; overflow:hidden; box-shadow:0 8px 28px rgba(14,124,134,.14); }
.report-head { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; background:#0e7c86; padding:.75rem 1.4rem; letter-spacing:.1em; font-size:.92rem; }
.report-head, .report-head * { color:#ffffff !important; }
.report-body { display:flex; align-items:center; gap:1.8rem; padding:1.7rem 1.6rem; flex-wrap:wrap; }
.ring { width:158px; height:158px; border-radius:50%; flex:none; display:grid; place-items:center;
        background:conic-gradient(var(--c) calc(var(--p) * 1%), #e1edf0 0); }
.ring-in { width:124px; height:124px; border-radius:50%; background:#fff; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.ring-in b { font-size:2.1rem; color:var(--c); line-height:1; }
.ring-in small { color:#5b7882; font-size:.8rem; margin-top:.2rem; }
.verdict h2 { margin:.4rem 0 .3rem; font-size:2.2rem; color:var(--c); }
.verdict p  { margin:.2rem 0; color:#35565f; font-size:1.08rem; }
.stamp { display:inline-block; border:3px solid var(--c); color:var(--c); border-radius:8px; padding:.15rem .9rem;
         font-weight:700; letter-spacing:.14em; transform:rotate(-4deg); font-size:.95rem; }
.high { --c:#d9363e; } .low { --c:#16955f; }
.high .report-body { background:linear-gradient(90deg,#fff5f5,#ffffff 55%); animation:beat 2s infinite; }
.low  .report-body { background:linear-gradient(90deg,#f1fbf6,#ffffff 55%); }
@keyframes beat { 0%,100%{transform:scale(1);} 50%{transform:scale(1.008);} }

/* ---------- Misc ---------- */
div[data-testid="stExpander"] { background:#fff; border:1px solid #cfe3e8; border-radius:12px; }
[data-testid="stAlert"] { border-radius:12px; }
.hosp-footer { text-align:center; color:#6b8790; margin-top:2rem; font-size:.95rem; letter-spacing:.04em; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Load Artifacts
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    scaler  = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    model   = joblib.load("KNN_Heart_Prediction.pkl")
    return scaler, list(columns), model

try:
    scaler, columns, model = load_resources()
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.stop()

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""<div class='hosp-header'>
<div class='brand'><span class='cross'></span><div>
<small>DEPARTMENT OF CARDIOLOGY · AI DIAGNOSTIC UNIT</small>
<h1>Heart Disease Prediction by Farhan Shahriyar</h1>
<p>Fill in the patient details below and click <b>Predict</b> to estimate heart disease risk.</p>
</div></div>
<svg class='ecg' viewBox='0 0 320 60' preserveAspectRatio='none'>
<path class='base' pathLength='400' d='M0 30H70L80 30L90 8L102 52L112 30H190L200 30L210 4L224 56L234 30H320'/>
<path class='pulse' pathLength='400' d='M0 30H70L80 30L90 8L102 52L112 30H190L200 30L210 4L224 56L234 30H320'/>
</svg></div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Input Form  (inputs unchanged)
# ------------------------------------------------------------------
with st.form("prediction_form"):
    st.markdown("### 📋 Patient Information")

    tab1, tab2, tab3 = st.tabs(
        ["👤 Basic Info", "📊 Clinical Measurements", "🫀 Test Results"]
    )

    # ---------- Tab 1 : Basic ----------
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age (years)", 20, 100, 50)
            sex = st.selectbox("Sex", ["M", "F"],
                               format_func=lambda x: "Male" if x == "M" else "Female")
        with c2:
            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["ATA", "NAP", "TA", "ASY"],
                format_func=lambda x: {
                    "ATA": "ATA — Atypical Angina",
                    "NAP": "NAP — Non-Anginal Pain",
                    "TA":  "TA  — Typical Angina",
                    "ASY": "ASY — Asymptomatic",
                }[x],
            )
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL", [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

    # ---------- Tab 2 : Measurements ----------
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 600, 200)
        with c2:
            max_hr  = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
            oldpeak = st.slider("Oldpeak (ST Depression)", -3.0, 7.0, 1.0, 0.1)

    # ---------- Tab 3 : Tests ----------
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"],
                format_func=lambda x: {
                    "Normal": "Normal",
                    "ST":  "ST-T Wave Abnormality",
                    "LVH": "Left Ventricular Hypertrophy",
                }[x],
            )
            exercise_angina = st.selectbox(
                "Exercise Induced Angina", ["N", "Y"],
                format_func=lambda x: "Yes" if x == "Y" else "No",
            )
        with c2:
            st_slope = st.selectbox(
                "ST Slope",
                ["Up", "Flat", "Down"],
                format_func=lambda x: {
                    "Up":   "Upsloping",
                    "Flat": "Flat",
                    "Down": "Downsloping",
                }[x],
            )

    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", width="stretch")

# ------------------------------------------------------------------
# Feature Engineering  (unchanged)
# ------------------------------------------------------------------
def build_feature_row() -> dict:
    return {
        "Age":                 age,
        "RestingBP":           resting_bp,
        "Cholesterol":         cholesterol,
        "FastingBS":           fasting_bs,
        "MaxHR":               max_hr,
        "Oldpeak":             oldpeak,
        "Sex_M":               1 if sex == "M" else 0,
        "ChestPainType_ATA":   1 if chest_pain == "ATA" else 0,
        "ChestPainType_NAP":   1 if chest_pain == "NAP" else 0,
        "ChestPainType_TA":    1 if chest_pain == "TA"  else 0,
        "RestingECG_Normal":   1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST":       1 if resting_ecg == "ST"     else 0,
        "ExerciseAngina_Y":    1 if exercise_angina == "Y"  else 0,
        "ST_Slope_Flat":       1 if st_slope == "Flat" else 0,
        "ST_Slope_Up":         1 if st_slope == "Up"   else 0,
    }

def report_card(kind, label, stamp, pct, caption, note):
    now = datetime.now().strftime("%d %b %Y · %I:%M %p")
    return f"""<div class='report {kind}'>
<div class='report-head'><span>CARDIOLOGY DIAGNOSTIC REPORT</span><span>{now}</span></div>
<div class='report-body'>
<div class='ring' style='--p:{pct:.1f}'><div class='ring-in'><b>{pct:.1f}%</b><small>{caption}</small></div></div>
<div class='verdict'><span class='stamp'>{stamp}</span><h2>{label}</h2><p>{note}</p></div>
</div></div>"""

# ------------------------------------------------------------------
# Prediction + Results
# ------------------------------------------------------------------
if submitted:
    row = build_feature_row()
    input_df = pd.DataFrame([row]).reindex(columns=columns, fill_value=0)
    scaled = scaler.transform(input_df)

    prediction = int(model.predict(scaled)[0])
    proba      = model.predict_proba(scaled)[0]

    st.markdown("---")
    st.markdown("## 🩺 Prediction Result")

    left, right = st.columns([1.2, 1])

    # ---- Result Card ----
    with left:
        if prediction == 1:
            st.markdown(report_card(
                "high", "⚠️ High Risk", "HIGH RISK", proba[1] * 100,
                "Disease probability",
                "Probability of Heart Disease. Please consult a cardiologist."),
                unsafe_allow_html=True)
        else:
            st.markdown(report_card(
                "low", "✅ Low Risk", "LOW RISK", proba[0] * 100,
                "No-disease probability",
                "Probability of No Heart Disease. Keep up regular check-ups."),
                unsafe_allow_html=True)

    # ---- Probabilities ----
    with right:
        st.markdown("#### 📊 Probability Breakdown")
        st.progress(float(proba[0]), text=f"✅ No Disease — {proba[0]*100:.1f}%")
        st.progress(float(proba[1]), text=f"⚠️ Disease — {proba[1]*100:.1f}%")

        st.markdown("#### 🧠 Model Info")
        st.info(
            f"**Algorithm:** K-Nearest Neighbors (k = {model.n_neighbors})\n\n"
            f"**Features used:** {len(columns)}"
        )

    # ---- Summary Table ----
    with st.expander("🔎 View Patient Input Summary", expanded=False):
        summary = pd.DataFrame({
            "Feature": [
                "Age", "Sex", "Chest Pain Type", "Resting BP",
                "Cholesterol", "Fasting BS > 120", "Resting ECG",
                "Max Heart Rate", "Exercise Angina", "Oldpeak", "ST Slope",
            ],
            "Value": [
                f"{age} years", "Male" if sex == "M" else "Female",
                chest_pain, f"{resting_bp} mm Hg", f"{cholesterol} mg/dL",
                "Yes" if fasting_bs == 1 else "No", resting_ecg,
                f"{max_hr} bpm", "Yes" if exercise_angina == "Y" else "No",
                f"{oldpeak:.1f}", st_slope,
            ],
        })
        st.dataframe(summary, width="stretch", hide_index=True)

    # ---- Disclaimer ----
    st.warning(
        "⚠️ **Medical Disclaimer:** This prediction is generated by a machine "
        "learning model and is for educational purposes only. It is **not** a "
        "substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider."
    )

else:
    st.info("👆 Fill in the patient details above and click **Predict** to see the result.")

st.markdown("<div class='hosp-footer'>✚ Cardiology AI Diagnostic Unit · Developed by Farhan Shahriyar</div>",
            unsafe_allow_html=True)