# pages/1_📊_Data_Overview.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import load_data  # Import from utils

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

st.title("📊 Data Overview")
st.markdown("Let's take a look at the data used to train our phishing detection models.")
st.markdown("---")

# --- Load Data ---
df = load_data() # Load data using the utility function

if df is not None:
    st.markdown("### Dataset Shape")
    st.write(f"The dataset contains **{df.shape[0]}** samples (URLs) and **{df.shape[1]}** features (including the label).")

    st.markdown("### Data Sample")
    st.dataframe(df.head())

    st.markdown("### Missing Values")
    missing_values = df.isnull().sum()
    missing_df = pd.DataFrame(missing_values[missing_values > 0], columns=['Missing Count'])
    if not missing_df.empty:
        st.warning("Missing values found:")
        st.dataframe(missing_df)
        # Basic imputation or warning
        st.info("Note: For simplicity, missing numerical values might be filled with 0 or median during training, or rows dropped.")
        # df = df.fillna(0) # Example: Simple fillna - move this logic to training step if needed
    else:
        st.success("✅ No missing values found in the dataset.")

    st.markdown("### Feature Explanations")
    st.info("Here are some of the 'clues' (features) extracted from URLs to help detect phishing:")

    # Example feature explanations using st.popover (modern alternative to tooltips)
    # Adjust feature names based on your actual 'urldata.csv' columns
    feature_explanations = {
        "Having_IP": "Checks if the URL's domain part is just an IP address (e.g., `192.168.1.1/login.html`). Phishers sometimes use IPs to bypass domain blacklists. (1 = Yes, 0 = No)",
        "Have_At": "Checks if the URL contains the '@' symbol. This can sometimes be used to obscure the actual domain. (1 = Yes, 0 = No)",
        "URL_Length": "Categorizes the URL length. Very long URLs can be suspicious. (e.g., 1 = Long >= 54 chars, 0 = Medium, -1 = Short - *adjust based on actual encoding*)",
        "URL_Depth": "Counts the number of '/' separators in the URL path (e.g., `example.com/a/b/c` has depth 3). Deep paths might hide malicious pages.",
        "Redirection": "Checks for '//' redirection operators within the URL path. Phishers might use this to redirect users sneakily. (1 = Yes, 0 = No)",
        "https_Domain": "Checks if the *string* 'https' appears within the *domain name itself* (e.g., `https-secure-login.com`). This is different from using the HTTPS protocol. (1 = Yes, 0 = No)",
        "TinyURL": "Checks if the domain belongs to a known URL shortening service (like bit.ly, t.co). While legitimate, phishers often use shorteners to hide the final destination. (1 = Yes, 0 = No)",
        "Prefix/Suffix": "Checks if the domain name contains a hyphen '-'. Legitimate sites use them, but phishers might add prefixes/suffixes to mimic real domains (e.g., `paypal-login.com`). (1 = Yes, 0 = No)",
        "Domain_Age": "The age of the domain in days, obtained via WHOIS lookup. Phishing sites are often very new (recently registered). (-1 if lookup failed or domain is very new).",
        # Add more features from your dataset here...
        "Label": "The target variable: 0 indicates a legitimate ('Benign') URL, and 1 indicates a phishing ('Malicious') URL."
    }

    # Display explanations dynamically based on columns present
    cols_to_explain = [col for col in feature_explanations if col in df.columns]
    for feature in cols_to_explain:
         with st.popover(f"ℹ️ {feature}"):
              st.markdown(f"**{feature}:** {feature_explanations[feature]}")
         # Add a small space or divider
         st.write("") # Creates a small vertical space

    st.markdown("_(Hover over the ℹ️ icons above for details on each feature.)_")


    st.markdown("### Dataset Summary")
    st.info(
        """
        This dataset provides a collection of URLs characterized by these structural and content-based features.
        The goal is to train models that can learn the patterns distinguishing legitimate URLs from phishing attempts
        based on these 'clues'. The `Label` column tells us the ground truth for each URL.
        """
    )
else:
     st.error("Failed to load data. Cannot display overview.")

st.markdown("---")