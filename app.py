import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

# Load model
model = joblib.load('best_model.pkl')

# Page Configuration
st.set_page_config(page_title="Performance Insights", layout="centered")

# Sidebar
with st.sidebar:
    st.title("🧠 Performance Insights")
    st.write("---")
    st.write("Machine Learning powered grade forecasting. Features: Study Habits, Wellness, and Part-Time Work impacts.")
    st.write("---")
    st.info("📈 Top Factor: Study Hours (82% correlation)")
    st.info("📈 2nd Factor: Mental Health Rating (32% correlation)")
    st.warning("📉 Weakest Factor: Part-Time Job (1% correlation)")
    st.success("📚 Trained on 1,000+ student records")

# Main Title
st.title("Student Performance Predictor")
st.write("")

# Two-Column Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Academic Factors", divider="gray")
    study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=8.3, value=2.0, step=0.5)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    part_time = st.selectbox("Part-Time Job", ["No", "Yes"])

with col2:
    st.subheader("Lifestyle Factors", divider="gray")
    mental_health = st.slider("Mental Health Rating (1-10)", min_value=1, max_value=10, value=5)
    sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=12.0, value=7.0, step=0.5)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction
if st.button("Predict Exam Score", use_container_width=True, type="primary"):

    part_time_val = 1 if part_time == "Yes" else 0

    input_features = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_val]])

    predicted_score = float(np.clip(model.predict(input_features)[0], 0, 100))

    st.markdown(f"<h3 style='text-align: center;'>Predicted Exam Score: {predicted_score:.1f}%</h3>", unsafe_allow_html=True)

    # Grade label
    if predicted_score >= 70:
        st.markdown("<p style='text-align:center; font-size:18px;'>🟢 Good — Keep it up!</p>", unsafe_allow_html=True)
    elif predicted_score >= 40:
        st.markdown("<p style='text-align:center; font-size:18px;'>🟡 Average — Room to improve.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='text-align:center; font-size:18px;'>🔴 At Risk — Consider seeking support.</p>", unsafe_allow_html=True)

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_score,
        number={'suffix': '%', 'valueformat': '.1f'},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkgray"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40],  'color': 'red'},
                {'range': [40, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'green'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
