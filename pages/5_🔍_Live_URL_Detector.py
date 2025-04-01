# pages/5_🔍_Live_URL_Detector.py
import streamlit as st
import pandas as pd
from utils import extract_features, load_data # Import helpers

st.set_page_config(page_title="Live URL Detector", page_icon="🔍", layout="wide")

st.title("🔍 Live Phishing URL Detector")
st.markdown("Enter a URL below to classify it using one of the trained models.")
st.markdown("---")

# --- Check for Models ---
if "model_results" not in st.session_state or not st.session_state["model_results"]:
    st.warning("⚠️ No models available. Please train or load models on the 'Train & Evaluate' page first.", icon="🚨")
    st.stop() # Stop execution if no models are loaded/trained

model_results = st.session_state["model_results"]
model_names = list(model_results.keys())

# --- Load Data Just to Get Feature Names ---
# This ensures the extracted features match what the model expects
df_detector = load_data()
expected_features = []
if df_detector is not None:
    # Prepare expected feature list (mirroring training)
    data_model_det = df_detector.copy()
    potential_drop = ['Domain', 'URL', 'id', 'Label']
    cols_to_drop = [col for col in potential_drop if col in data_model_det.columns]
    if cols_to_drop:
        data_model_det = data_model_det.drop(columns=cols_to_drop)
    if data_model_det.isnull().sum().sum() > 0:
         data_model_det = data_model_det.fillna(0)
    expected_features = list(data_model_det.columns)


if not expected_features:
     st.error("❌ Could not determine the expected features for the models. Cannot proceed with live detection.")
     st.stop()


# --- User Input ---
col1, col2 = st.columns([3, 1])
with col1:
    url_input = st.text_input("Enter the full URL to check (e.g., `http://example.com`):", key="url_input")
with col2:
    selected_model_name = st.selectbox("Select Model:", model_names, key="detector_model_select",
                                      help="Choose which trained model to use for prediction.")

# --- Prediction Button ---
if st.button("Analyze URL", key="analyze_button"):
    if not url_input or not url_input.strip():
        st.warning("⚠️ Please enter a URL to analyze.", icon="❗")
    elif not selected_model_name:
         st.warning("⚠️ Please select a model.", icon="❗")
    else:
        st.markdown("---")
        with st.spinner(f"Analyzing URL with {selected_model_name}... This might take a moment (especially if domain age lookup is enabled)."):
            # Extract features using the utility function
            features_df = extract_features(url_input, expected_features)

            if features_df is not None:
                # Get the selected model object
                model_obj = model_results[selected_model_name]["model_object"]

                # Predict
                try:
                    prediction = model_obj.predict(features_df)
                    probability = -1.0 # Default if predict_proba not available
                    if hasattr(model_obj, "predict_proba"):
                         proba = model_obj.predict_proba(features_df)
                         # Probability of being phishing (class 1)
                         probability = proba[0][1]

                    # Display Result
                    st.subheader("🚦 Analysis Result")
                    if prediction[0] == 1: # Phishing
                        st.error(f"🚨 **Phishing Alert!**", icon="🎣")
                        st.markdown(f"The selected model (`{selected_model_name}`) classifies this URL as **likely malicious**.")
                        if probability >= 0:
                             st.markdown(f"Confidence Score (Phishing): **{probability:.2%}**")
                        st.warning("Exercise extreme caution. Do not enter credentials or sensitive information.", icon="⚠️")
                    else: # Legitimate
                        st.success(f"✅ **Likely Safe**", icon="🛡️")
                        st.markdown(f"The selected model (`{selected_model_name}`) classifies this URL as **likely legitimate**.")
                        if probability >= 0:
                             st.markdown(f"Confidence Score (Phishing): **{probability:.2%}** (Lower is better)")
                        st.info("While likely safe according to the model, always remain vigilant online.", icon="💡")

                    # Optional: Display Extracted Features
                    with st.expander("View Extracted Features Used for Prediction"):
                        st.dataframe(features_df.style.format("{:.3f}", na_rep="-"))
                        st.caption("Features extracted from the URL. Values might be 0/1 flags, counts, scores, or -1 if unavailable (like Domain Age).")

                except Exception as e:
                    st.error(f"❌ An error occurred during prediction: {e}")
            else:
                st.error("❌ Failed to extract features from the URL. Cannot make a prediction.")
        st.markdown("---")

# --- Explanation ---
st.sidebar.title("ℹ️ How it Works")
st.sidebar.info(
    """
    1.  You enter a URL.
    2.  The app extracts relevant features (like `Having_IP`, `URL_Length`, `Domain_Age`, etc.) based on the patterns learned during training.
    3.  The selected Machine Learning model analyzes these features.
    4.  The model predicts whether the URL is `Phishing (1)` or `Legit (0)`.
    5.  The result is displayed along with the model used.
    """
)
st.sidebar.warning("**Disclaimer:** Automated tools provide valuable insights but are not foolproof. Always use critical thinking when Browse.")