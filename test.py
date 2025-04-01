import sys
import os
import pandas as pd
import pickle

# Ensure app.py is in the path to import functions
sys.path.append(os.path.dirname(os.path.abspath("app.py")))

# Import extract_features from app.py
from app import extract_features

# Load trained models
MODEL_FILE = "trained_models.pkl"
if not os.path.exists(MODEL_FILE):
    print(f"[ERROR] Model file '{MODEL_FILE}' not found. Train models first.")
    sys.exit(1)

with open(MODEL_FILE, "rb") as f:
    models = pickle.load(f)

# Load verified URLs
CSV_FILE = "verified_online.csv"
if not os.path.exists(CSV_FILE):
    print(f"[ERROR] CSV file '{CSV_FILE}' not found.")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)

# Randomly sample a subset from the CSV (adjust subset_size as needed)
subset_size = 20
df_subset = df.sample(n=subset_size, random_state=42)

url_column = df_subset.columns[1]  # Assuming the second column has URLs
urls = df_subset[url_column].dropna().astype(str).tolist()

# Extract features for each URL in the subset
print(f"[INFO] Extracting features for {len(urls)} URLs from the random subset...")

feature_list = []
for i, url in enumerate(urls):
    try:
        features = extract_features(url)
        print(f"[DEBUG] Extracted {len(features)} features for {url}")
        feature_list.append(features)
    except Exception as e:
        print(f"[ERROR] Failed to extract features for {url}: {e}")
        feature_list.append([None] * 19)  # Fill with None if extraction fails

# Convert features to DataFrame
feature_names = [
    "Have_IP", "Have_At", "URL_Length", "URL_Depth", "Redirection",
    "https_Domain", "TinyURL", "Prefix/Suffix", "URL_Entropy", "Domain_Entropy",
    "Subdomain_Count", "Digit_Count", "Special_Char_Count", "Uppercase_Ratio",
    "Domain_Age", "iFrame", "Mouse_Over", "Right_Click", "Web_Forwards"
]
features_df = pd.DataFrame(feature_list, columns=feature_names)

# Remove 'Domain' column if it exists (models expect numeric data)
if "Domain" in features_df.columns:
    features_df.drop(columns=["Domain"], inplace=True)

# Run predictions using the best model (XGBoost)
best_model_name = "XGBoost"
if best_model_name not in models:
    print(f"[ERROR] Model '{best_model_name}' not found in trained models.")
    sys.exit(1)

best_model = models[best_model_name]
predictions = best_model.predict(features_df)

# Save predictions back into the subset DataFrame
df_subset["Prediction"] = predictions
df_subset["Prediction_Label"] = df_subset["Prediction"].map({1: "Phishing", 0: "Legit"})

OUTPUT_FILE = "verified_results_subset.csv"
df_subset.to_csv(OUTPUT_FILE, index=False)
print(f"[INFO] Predictions saved to {OUTPUT_FILE}")