import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration (Must be the very first command)
st.set_page_config(
    page_title="ChurnGuard | AI Retention",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS to make it look "Exotic"
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 15px;
        font-size: 20px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: 2px solid #ff4b4b;
    }
    h1 {
        color: #1f2937;
    }
    h2, h3 {
        color: #374151;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model and Scaler (Cached for performance)
@st.cache_resource
def load_assets():
    # Make sure these files are in your GitHub repo!
    model = tf.keras.models.load_model('churn_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("⚠️ Error loading model. Please check that 'churn_model.h5' and 'scaler.pkl' are in the GitHub repository.")
    st.stop()

# 4. Sidebar Section
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("ChurnGuard AI")
    st.markdown("---")
    st.info("This Deep Learning tool helps bank managers identify customers at risk of leaving.")
    st.markdown("### ⚙️ Model Details")
    st.code("Architecture: ANN\nFramework: TensorFlow\nAccuracy: ~85%", language="text")
    st.markdown("---")
    st.write("Designed for: **Deep Learning for Managers**")

# 5. Main Dashboard Interface
st.title("🏦 Customer Retention Intelligence")
st.markdown("### 🔍 Predict & Analyze Churn Risk")
st.write("Adjust the customer parameters below to simulate different scenarios.")
st.markdown("---")

# Layout: 3 Columns for organized inputs
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 92, 35)
    location = st.selectbox("Geography", ["France", "Germany", "Spain"])

with col2:
    st.subheader("💳 Financials")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 60000.0, step=1000.0)
    estimated_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 50000.0, step=1000.0)

with col3:
    st.subheader("🤝 Relationship")
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    num_of_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True)
    is_active_member = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True)

st.markdown("---")

# 6. Prediction Logic
if st.button("🚀 Analyze Risk Probability"):
    
    # 6.1 Preprocessing (Must match the training logic exactly)
    gen_val = 1 if gender == "Male" else 0
    cr_val = 1 if has_cr_card == "Yes" else 0
    act_val = 1 if is_active_member == "Yes" else 0
    
    # One-Hot Encoding for Geography
    geo_ger = 1 if location == "Germany" else 0
    geo_spn = 1 if location == "Spain" else 0
    
    # Create the DataFrame in the EXACT order the model expects
    input_data = pd.DataFrame([[credit_score, gen_val, age, tenure, balance, 
                                num_of_products, cr_val, act_val, estimated_salary, 
                                geo_ger, geo_spn]],
                              columns=['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                                       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
                                       'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'])
    
    # Scale the data
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    churn_prob = float(prediction[0][0])
    
    # 7. Results Display
    st.markdown("### 📊 Analysis Result")
    
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        # Visual Metric
        if churn_prob > 0.5:
            st.metric(label="Churn Risk", value="HIGH RISK", delta=f"{churn_prob:.1%}", delta_color="inverse")
        else:
            st.metric(label="Churn Risk", value="SAFE", delta=f"{churn_prob:.1%}", delta_color="normal")
            
    with result_col2:
        # Progress Bar and Interpretation
        st.write("#### Probability Gauge")
        st.progress(churn_prob)
        
        if churn_prob > 0.5:
            st.error(f"⚠️ **Alert:** This customer has a **{churn_prob*100:.1f}%** probability of leaving.")
            st.markdown("**Managerial Action:** Consider offering a **retention bonus** or checking for recent service complaints.")
        else:
            st.success(f"✅ **Safe:** This customer is likely to stay (**{100 - churn_prob*100:.1f}%** loyalty probability).")
            st.markdown("**Managerial Action:** No immediate action needed. Maintain standard service quality.")