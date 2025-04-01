# 🏠_Home.py
import streamlit as st
import pandas as pd
from utils import load_data  # Import helper from utils.py

# --- Page Configuration ---
st.set_page_config(
    page_title="Phishing Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Page Content ---
st.title("🛡️ Phishing URL Detector Pro")
st.markdown("---")

st.subheader("🏠 Welcome!")

# --- Problem Statement ---
st.markdown("#### The Problem: The Danger of Phishing 🎣")
st.info(
    """
    Phishing attacks are a common and dangerous cyber threat. Malicious actors create fake websites
    that mimic legitimate ones (like banks, email providers, or social media) to trick users into
    revealing sensitive information like passwords, credit card numbers, or personal details.
    These attacks can lead to identity theft, financial loss, and unauthorized access to accounts.
    Recognizing these fake URLs quickly is crucial for online safety.
    """
)

# --- App Objective ---
st.markdown("#### Our Objective: AI-Powered Detection 🧠")
st.success(
    """
    This application uses Machine Learning (ML) to automatically detect phishing URLs.
    By analyzing various technical features extracted from a URL (like its structure, domain age,
    use of HTTPS, etc.), our trained models can classify whether a link is likely
    **legitimate (`✅ Safe`)** or **malicious (`🚨 Phishing`)**.
    """
)

# --- App Overview ---
st.markdown("#### App Structure: What Can You Do Here? 🗺️")
st.markdown(
    """
    This multi-page application guides you through the process:

    1.  **📊 Data Overview:** Explore the dataset used to train our models. See its shape, a sample,
        and learn about the features (the 'clues') we use.
    2.  **📈 EDA (Exploratory Data Analysis):** Visualize the data patterns. See how different features
        are distributed and how they correlate with each other.
    3.  **🛠️ Model Training & Evaluation:** Train different ML models (like Decision Trees, Random Forests)
        or load pre-trained ones. See how well they perform using metrics like accuracy and precision.
    4.  **⚖️ Model Comparison:** Compare the performance of all trained models side-by-side to see
        which one works best for this task and understand *why* based on feature importance.
    5.  **🔍 Live URL Detector:** Test any URL in real-time! Paste a link, and the app will use a
        trained model to predict if it's safe or phishing.

    Use the sidebar navigation on the left to explore each section!
    """
)
st.markdown("---")
st.warning("⚠️ **Disclaimer:** This tool provides predictions based on patterns learned from data. Always exercise caution when clicking links or entering sensitive information online. No detection tool is 100% perfect.")

# --- Optional: Display small sample from data ---
st.markdown("#### Quick Peek at the Data")
try:
    df = load_data()
    st.dataframe(df.head(3))
except FileNotFoundError:
    st.error("Error: `urldata.csv` not found. Please ensure the data file is in the same directory.")
except Exception as e:
    st.error(f"An error occurred loading the data: {e}")