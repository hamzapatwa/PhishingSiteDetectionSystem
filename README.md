# 🛡️ Phishing URL Detector

## Overview
**Phishing URL Detector** is a multi-page Streamlit dashboard that leverages machine learning to detect phishing websites based on structural, lexical, WHOIS, and HTML/JS features. This tool empowers users to explore how phishing detection works—from data analysis and model training to real-time predictions.

> **Why it matters:** Phishing attacks are one of the most prevalent cyber threats. Automating URL classification using ML helps reduce risk at scale and enhances proactive defense mechanisms.

---

## 🔄 What’s New
- ✅ **Refactored to a multi-page Streamlit app** for better modularity and UX.
- ⚙️ Optimized WHOIS domain age feature with fallbacks.
- 📊 Enhanced EDA and feature visualization.
- 🔍 Real-time URL detector with improved feature extraction.

---

## ⚙️ Features & Capabilities
- 🔬 Live phishing URL prediction using trained ML models.
- 📊 Dataset exploration and visual summaries.
- 📈 Feature correlation and distribution visualizations.
- 🛠️ Train custom models (Decision Tree, Random Forest, MLP, XGBoost).
- ⚖️ Compare models using accuracy, precision, and confusion matrices.
- 💾 Load pre-trained models for fast testing.

---

## 📁 Dataset
- **Source**: Public phishing and legitimate URL datasets.
- **Structure**: CSV format with ~30 extracted features per URL.
- **Target Label**: `Label` column (0 = Legitimate, 1 = Phishing)

---

## 🧪 Feature Engineering
Features are grouped into:

### 🔗 Address Bar-Based Features
- `Having_IP`
- `Have_At`
- `URL_Length`
- `URL_Depth`
- `Redirection`
- `https_Domain`
- `TinyURL`
- `Prefix/Suffix`

### 🧠 WHOIS-Based Features
- `Domain_Age`
- `Domain_registration_length`

### 💻 HTML & JS Features
- `iFrame`
- `Mouse_Over`
- `Right_Click`
- `Web_Forwards`

### 📊 Lexical Features
- `Subdomain_Count`
- `Entropy`

> Missing or unavailable values are handled with default fallbacks (e.g., -1).

---

## 🧠 Model Training & Evaluation
Train or load the following models:

- **Decision Tree**
- **Random Forest**
- **MLP (Neural Net)**
- **XGBoost**

### 🔍 Evaluation Metrics:
- Accuracy
- Precision
- Confusion Matrix (Plotly heatmap)
- Feature Importance (Bar charts for tree-based models)

---

## 💾 Model Saving & Loading
```python
# Save trained models
to_save = {'XGBoost': xgb_model, 'Random Forest': rf_model}
with open('trained_models.pkl', 'wb') as f:
    pickle.dump(to_save, f)

# Load models in app
with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)
```

---

## 📁 Multi-Page App Navigation
### Structure:
```
.
├── 🏠_Home.py               # Main app entry point
├── utils.py                    # Helper functions (loaders, plots, trainers)
├── trained_models.pkl          # Pre-trained ML models
├── urldata.csv                 # Dataset
├── requirements.txt
└── pages/
    ├── 1_📊_Data_Overview.py
    ├── 2_📈_EDA.py
    ├── 3_🛠️_Model_Training_&_Evaluation.py
    ├── 4_⚖️_Model_Comparison.py
    └── 5_🔍_Live_URL_Detector.py
```

---

## 🚀 Setup & Installation
```bash
# Clone repo or copy files
pip install -r requirements.txt  # Install Python dependencies

# Run the Streamlit app
streamlit run 🏠_Home.py
```
> Ensure `graphviz` is installed on your system:
> - macOS: `brew install graphviz`
> - Ubuntu: `sudo apt install graphviz`

---

## 📌 Future Work
- 🧠 Add deep learning models (LSTM for sequence-based analysis).
- 🔗 Integrate threat intelligence feeds for real-time blacklisting.
- 📬 Email scanner plugin or browser extension.
- 🌐 Host as a public web app with authentication.
- 📈 More advanced visual analytics (SHAP, LIME).

---

## 📚 License & References
**License**: MIT

**References**:
- [PhishTank Dataset](https://www.phishtank.com/)
- [WHOIS Python Module](https://pypi.org/project/python-whois/)
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [XGBoost](https://xgboost.readthedocs.io/)

---

> 🚧 Disclaimer: This tool is for educational and experimental use only. Do not rely on it for production-grade security decisions.

