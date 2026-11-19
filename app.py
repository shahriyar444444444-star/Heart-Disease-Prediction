import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Page Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Custom CSS (Dark Theme & Interactive UI)
# ------------------------------------------------------------------
st.markdown("""
<style>
/* Times New Roman Font Applied Globally */
html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, div, span, button, label, table, th, td { 
    font-family: 'Times New Roman', Times, serif !important; 
}

/* Dark Gradient Background */
.stApp { 
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); 
    color: #ffffff;
}

/* Header Card */
.main-header {
    text-align: center; padding: 1.8rem;
    background: rgba(255, 255, 255, 0.05); /* Transparent dark card */
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px; margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.main-header h1 { color: #ff4b4b; font-size: 2.8rem; margin: 0; text-shadow: 0 0 15px rgba(255,75,75,0.4); }
.main-header p  { color: #d1d1d1; margin-top: 0.4rem; font-size: 1.1rem; }

/* Form Container styling */
[data-testid="stForm"] {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* Labels and Text */
.stSelectbox label, .stSlider label, h3, h4, p { 
    color: #f0f0f0 !important; 
    font-weight: 600;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] { 
    gap: 10px; 
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255, 255, 255, 0.08); 
    border-radius: 25px; /* বক্সটিকে গোলাকার (Rounded Square) করার জন্য */
    padding: 10px 24px; /* প্যাডিং অ্যাডজাস্ট করা হয়েছে */
    color: #b0b0b0;
    border: 1px solid transparent;
    font-size: 1.1rem;
    display: flex; 
    justify-content: center; /* লেখাকে ঠিক মাঝখানে আনার জন্য */
    align-items: center; /* লেখা ও আইকন ঠিক উচ্চতায় রাখার জন্য */
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(255, 75, 75, 0.15) !important;
    color: #ff4b4b !important;
    border: 1px solid #ff4b4b;
}

/* Gradient Button */
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b 0%, #f09819 100%);
    color: white; border: none; padding: 0.85rem 2rem;
    font-size: 1.3rem; border-radius: 12px; font-weight: 700;
    width: 100%; transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(240, 152, 25, 0.4);
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(240, 152, 25, 0.6);
}

/* Result Cards */
.result-card-high {
    background: linear-gradient(135deg, #ff4b4b 0%, #b91212 100%);
    padding: 2.5rem; border-radius: 20px; text-align: center;
    color: white; box-shadow: 0 12px 40px rgba(255,75,75,0.4);
    animation: pulse 2s infinite;
}
.result-card-low {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    padding: 2.5rem; border-radius: 20px; text-align: center;
    color: white; box-shadow: 0 12px 40px rgba(0,176,155,0.4);
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

/* Expander Background */
div[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.05); 
    border-radius: 12px; 
    border: 1px solid rgba(255, 255, 255, 0.1);
}
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
st.markdown("""
<div class='main-header'>
    <h1>Heart Disease Prediction by Farhan Shahriyar</h1>
    <p>Fill in the patient details below and click <b>Predict</b> to estimate heart disease risk.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Input Form
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

    # Using width="stretch" instead of use_container_width to avoid deprecation
    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", width="stretch")

# ------------------------------------------------------------------
# Feature Engineering
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
            st.markdown(f"""
            <div class='result-card-high'>
                <h1 style='margin:0;font-size:2.4rem;'>⚠️ High Risk</h1>
                <p style='font-size:3.6rem;font-weight:700;margin:0.4rem 0;'>
                    {proba[1]*100:.1f}%
                </p>
                <p style='font-size:1.1rem;margin:0;'>Probability of Heart Disease</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-low'>
                <h1 style='margin:0;font-size:2.4rem;'>✅ Low Risk</h1>
                <p style='font-size:3.6rem;font-weight:700;margin:0.4rem 0;'>
                    {proba[0]*100:.1f}%
                </p>
                <p style='font-size:1.1rem;margin:0;'>Probability of No Heart Disease</p>
            </div>
            """, unsafe_allow_html=True)

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
        # Using width="stretch"
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