import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from google import genai




# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Customer Segmentation",
    layout="wide"
)

st.title("Customer Segmentation")




# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "..", "notebooks", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "notebooks", "processed")

MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "rfm_scaler.pkl")
DATA_PATH = os.path.join(DATA_DIR, "customer_status_enriched.csv")

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    kmeans = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return kmeans, scaler

kmeans, scaler = load_models()

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ================= STANDARDIZE COLUMN NAMES =================
df.rename(columns={
    "recency": "Recency",
    "frequency": "Frequency",
    "monetary": "Monetary"
}, inplace=True)

# ================= ENSURE CUSTOMER ID EXISTS =================
if "customer_id" not in df.columns:
    df["customer_id"] = df.index + 1
    
    
# ================= LLM CONFIG =================
USE_LLM = True

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    LLM_AVAILABLE = True
except:
    LLM_AVAILABLE = False
    
def llm_explanation_gemini(status, Recency, Frequency, Monetary, cluster):
    prompt = f"""
You are a customer analytics expert.

Customer Details:
- Recency: {Recency} days
- Frequency: {Frequency}
- Monetary Value: ₹{Monetary}
- Cluster: {cluster}
- Segment: {status}

Explain WHY this customer belongs to this segment.
Give ONE business recommendation.
Use simple language.
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except:
        return None


# ================= KPI FUNCTION =================
def show_kpis(data):
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", len(data))
    col2.metric("Avg Monetary", round(data["Monetary"].mean(), 2))
    col3.metric("Avg Frequency", round(data["Frequency"].mean(), 2))
    col4.metric("Avg Recency (Days)", round(data["Recency"].mean(), 2))

# ================= CUSTOMER STATUS LOGIC =================
def customer_status_logic(Recency, Frequency, Monetary):
    if Recency < 30 and Frequency >= 10 and Monetary > 3000:
        return "Very Happy & Loyal"
    elif Recency < 60 and Frequency >= 5:
        return "Happy & Active"
    elif Recency > 120:
        return "Churn Risk"
    elif Recency > 90:
        return "At Risk"
    else:
        return "Regular Customer"

# ================= AI EXPLANATION =================
def llm_explanation(status, Recency, Frequency, Monetary):
    return (
        f"This customer is classified as **{status}** because:\n\n"
        f"- Last purchase was **{Recency} days ago**\n"
        f"- Total orders placed: **{Frequency}**\n"
        f"- Total spending: **₹{Monetary}**\n\n"
        f"This indicates the customer's engagement and value."
    )

# ================= TABS =================
tab1, tab2, tab3 = st.tabs([
    "KPI Dashboard",
    "Real-Time Prediction",
    "Customer Lookup"
])

# ================= TAB 1: KPI DASHBOARD =================
with tab1:
    st.subheader(" Customer Health Overview")
    show_kpis(df)

    st.subheader("Customer Status Distribution")
    st.bar_chart(df["customer_status"].value_counts())

    st.subheader("Revenue by Customer Status")
    revenue = df.groupby("customer_status")["Monetary"].sum()
    st.bar_chart(revenue)

# ================= TAB 2: REAL-TIME PREDICTION =================
with tab2:
    st.subheader("Predict Customer Segment")

    Recency = st.number_input(
        "Recency (days since last purchase)", 0, 500, 30
    )
    Frequency = st.number_input(
        "Frequency (number of orders)", 0, 100, 5
    )
    Monetary = st.number_input(
        "Monetary Value (₹)", 0.0, 100000.0, 2000.0
    )

    if st.button("Predict"):
        input_data = np.array([[Recency, Frequency, Monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = int(kmeans.predict(input_scaled)[0])

        status = customer_status_logic(Recency, Frequency, Monetary)
        explanation = None

        if USE_LLM and LLM_AVAILABLE:
            explanation = llm_explanation_gemini(
                status, Recency, Frequency, Monetary, cluster
           )
        st.markdown("### AI Explanation")


        if explanation:
            st.markdown(explanation)
            st.caption("Generated using LLM")
        else:
            st.markdown(llm_explanation(status, Recency, Frequency, Monetary))
            st.caption("Rule-based explanation (fallback)")

        st.success(f"Predicted Cluster: {cluster}")
        st.info(f"Customer Status: {status}")


# ================= TAB 3: CUSTOMER LOOKUP =================
with tab3:
    st.subheader("Search Existing Customer")

    customer_id = st.selectbox(
        "Select Customer ID",
        df["customer_id"].unique()
    )

    cust = df[df["customer_id"] == customer_id].iloc[0]

    st.markdown("### Customer Summary")
    st.write({
        "Customer ID": cust["customer_id"],
        "Recency": cust["Recency"],
        "Frequency": cust["Frequency"],
        "Monetary": cust["Monetary"],
        "Customer Status": cust["customer_status"],
        "Recommended Action": cust.get("recommended_action", "N/A")
    })

