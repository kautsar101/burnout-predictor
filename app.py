import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Burnout Prediction System",
    page_icon="📊",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("catboost_burnout_model.pkl")
    scaler = joblib.load("scaler_burnout.pkl")
    return model, scaler

model, scaler = load_artifacts()

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.metric-card {
    background-color: #f8fafc;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid #e5e7eb;
}

.score-text {
    font-size: 48px;
    font-weight: bold;
    color: #1f2937;
}

.low-risk {
    background-color: #dcfce7;
    color: #166534;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

.medium-risk {
    background-color: #fef3c7;
    color: #92400e;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}

.high-risk {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.title("Work From Home Burnout Prediction System")

st.markdown("""
Predict employee burnout score based on work habits,
screen exposure, workload, and wellbeing indicators.
""")

st.divider()

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Work Environment")

    day_type = st.selectbox(
        "Day Type",
        ["Weekday", "Weekend"]
    )

    work_hours = st.slider(
        "Work Hours",
        min_value=0.0,
        max_value=16.0,
        value=8.0,
        step=0.5
    )

    screen_time_hours = st.slider(
        "Screen Time Hours",
        min_value=0.0,
        max_value=20.0,
        value=8.0,
        step=0.5
    )

    meetings_count = st.number_input(
        "Meetings Count",
        min_value=0,
        max_value=30,
        value=3
    )

with col2:
    st.subheader("Personal Wellbeing")

    breaks_taken = st.number_input(
        "Breaks Taken",
        min_value=0,
        max_value=20,
        value=3
    )

    after_hours_work = st.selectbox(
        "After Hours Work",
        ["No", "Yes"]
    )

    sleep_hours = st.slider(
        "Sleep Hours",
        min_value=0.0,
        max_value=12.0,
        value=7.0,
        step=0.5
    )

    task_completion_rate = st.slider(
        "Task Completion Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

st.divider()

# =========================
# PREDICTION
# =========================
if st.button("Predict Burnout Score", use_container_width=True):

    day_type_encoded = 0 if day_type == "Weekday" else 1
    after_hours_encoded = 1 if after_hours_work == "Yes" else 0

    input_df = pd.DataFrame({
        "day_type": [day_type_encoded],
        "work_hours": [work_hours],
        "screen_time_hours": [screen_time_hours],
        "meetings_count": [meetings_count],
        "breaks_taken": [breaks_taken],
        "after_hours_work": [after_hours_encoded],
        "sleep_hours": [sleep_hours],
        "task_completion_rate": [task_completion_rate]
    })

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    prediction = max(0, min(100, prediction))

    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div class="metric-card">
            <div>Burnout Score</div>
            <div class="score-text">{prediction:.2f}</div>
            <div>/ 100</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if prediction < 40:
        st.markdown(
            '<div class="low-risk">LOW RISK</div>',
            unsafe_allow_html=True
        )

    elif prediction < 70:
        st.markdown(
            '<div class="medium-risk">MEDIUM RISK</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            '<div class="high-risk">HIGH RISK</div>',
            unsafe_allow_html=True
        )

st.divider()

# =========================
# MODEL INFO
# =========================
with st.expander("Model Information"):

    st.write("Model : CatBoost Regressor")
    st.write("Target : Burnout Score (0 - 100)")

    st.write("Input Features:")

    st.write("""
    - Day Type
    - Work Hours
    - Screen Time Hours
    - Meetings Count
    - Breaks Taken
    - After Hours Work
    - Sleep Hours
    - Task Completion Rate
    """)