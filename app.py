import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib

# Load the model directly using the same library you saved it with
model = joblib.load('best_model.pkl')

# 1. Page Configuration
st.set_page_config(page_title="Performance Insights", layout="centered")

# 2. LOAD THE MODEL (THIS FIXES YOUR ERROR)
# Make sure 'best_model.pkl' is in the exact same folder as app.py


# ... rest of your code (sidebar, layout, etc.) ...

# 2. Add the Context Sidebar
with st.sidebar:
    st.title("🧠 Performance Insights")
    st.write("---")
    st.write("Machine Learning powered grade forecasting. Features: Study Habits, Wellness, and Part-Time Work impacts.")
    st.write("---")
    # Put your actual model accuracy here from your ipynb notebook
    st.metric(label='Model Accuracy', value='91.5%', delta='Optimized') 

# 3. Main Title
st.title("Student Performance Predictor")
st.write("") # Adds a little breathing room

# 4. The Two-Column Layout (Breaking the vertical scroll)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Academic Factors", divider="gray")
    # Changed from sliders to number_inputs for precision
    study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
    
    # Moved Part-Time here to balance the columns
    part_time = st.selectbox("Part-Time Job", ["No", "Yes"])

with col2:
    st.subheader("Lifestyle Factors", divider="gray")
    # Kept slider for subjective data, added number_input for sleep
    mental_health = st.slider("Mental Health Rating (1-10)", min_value=1, max_value=10, value=5)
    sleep_hours = st.number_input("Sleep Hours (per night)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)

st.markdown("<br>", unsafe_allow_html=True)

# 5. Prediction Button & Output
if st.button("Predict Exam Score", use_container_width=True, type="primary"):
    
    # --- YOUR PREDICTION LOGIC GOES HERE ---
    # Example (adjust based on how your model expects the data):
    # part_time_val = 1 if part_time == "Yes" else 0
    # features = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_val]])
    # predicted_score = model.predict(features)[0]
    
    # Placeholder for UI testing. Replace this variable with your model's actual output.
    # 1. Convert the dropdown to binary (assuming your model used 1 and 0 for Yes/No)
    part_time_val = 1 if part_time == "Yes" else 0
    
    # 2. Feed the features to the model in the EXACT order from your training data:
    # study -> attendance -> mental_health -> sleep -> part_time
    input_features = np.array([[study_hours, attendance, mental_health, sleep_hours, part_time_val]])
    
    # 3. Make the actual prediction using your loaded best_model.pkl
    predicted_score = model.predict(input_features)[0] 

    st.markdown(f"<h3 style='text-align: center;'>Predicted Exam Score: {predicted_score:.0f}%</h3>", unsafe_allow_html=True) 

    st.markdown(f"<h3 style='text-align: center;'>Predicted Exam Score: {predicted_score:.0f}%</h3>", unsafe_allow_html=True)
    
    # 6. The Plotly Gauge Chart (The "Main Event" for the examiner)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = predicted_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkgray"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'red'},     # Fail zone
                {'range': [40, 70], 'color': 'yellow'}, # Average zone
                {'range': [70, 100], 'color': 'green'}  # Good zone
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
