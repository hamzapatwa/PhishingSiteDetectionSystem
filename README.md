# Phishing URL Detection System

## Overview
This project aims to identify phishing websites using machine learning models trained on extracted URL features. Phishing websites mimic legitimate ones to steal sensitive information from users. Our system uses advanced feature extraction techniques, machine learning models, and a **Streamlit dashboard** for analysis and real-time URL checking.

## Features & Capabilities
- **Large-Scale Dataset**: Now trained on **100,000 URLs** (50,000 phishing, 50,000 legitimate).
- **Feature Extraction**: Analyzes URLs for address bar, domain, HTML/JavaScript, lexical, and statistical features.
- **Machine Learning Models**: Implements Decision Trees, Random Forests, XGBoost, SVM, and Multilayer Perceptrons (MLP).
- **Model Persistence**: Stores and loads trained models, eliminating the need for retraining on every launch.
- **Parallel Processing**: Utilizes **multithreading** for **faster HTTP-based feature extraction** and **WHOIS lookups**.
- **Efficient URL Processing**: Optimized **batch processing** of URLs for large-scale phishing detection.
- **Model Performance**: Evaluates models using accuracy, precision, and confusion matrices.
- **Visualization & Analysis**: Provides detailed **Exploratory Data Analysis (EDA)** and feature importance ranking.
- **Real-time Phishing Detection**: Accepts user input and classifies URLs as **phishing or legitimate**.
- **Streamlit Dashboard**: User-friendly interface for **model evaluation** and **live URL analysis**.

---

## Dataset
The dataset contains **100,000 URLs** from:
- **PhishTank**: A repository of verified phishing URLs.
- **University of New Brunswick**: A dataset of legitimate and phishing URLs.
- **Additional Sources**: Integration of more diverse phishing and legitimate URL sources for improved robustness.

### Feature Engineering
We extract several features from each URL.

#### **Address Bar Features:**
- **Having_IP**: Whether the URL contains an IP instead of a domain.
- **Have_At**: Presence of '@' symbol.
- **URL_Length**: URLs longer than 54 characters are suspicious.
- **URL_Depth**: Count of '/' in the path.
- **Redirection**: Checks occurrences of '//' beyond protocol.
- **HTTPS_Domain**: Detects if 'https' appears in the domain.
- **TinyURL**: Identifies usage of link-shortening services.
- **Prefix_Suffix**: Checks for '-' in domain names.

#### **Lexical & Statistical Features:**
- **URL Entropy**: Measures randomness in the URL.
- **Domain Entropy**: Measures randomness in the domain name.
- **Subdomain Count**: Counts additional subdomains.
- **Digit Count**: Number of numerical characters in the URL.
- **Special Character Count**: Identifies non-alphanumeric symbols.
- **Uppercase Ratio**: Percentage of uppercase letters.
- **Domain Age**: Computes domain registration age via WHOIS.
- **Character Frequency Analysis**: Tracks patterns of character distributions in phishing vs. legitimate URLs.

#### **HTML & JavaScript Features:**
- **iFrame**: Detects iframe redirection abuse.
- **Mouse Over**: Checks for JavaScript altering links.
- **Right Click Disable**: Detects websites preventing right-click actions.
- **Web Forwarding**: Counts redirections on the site.

### **Optimized Feature Extraction**
- **Threading for HTTP Requests**: HTTP-based features (iFrame, right-click detection, web forwarding, etc.) now use **multithreading** for significant performance improvements.
- **WHOIS Lookup Parallelization**: WHOIS lookups are now executed in parallel threads, reducing latency.
- **Feature Standardization**: Ensures consistency in feature values across different datasets.
- **URL Batch Processing**: Allows for processing multiple URLs simultaneously, reducing overall detection time.

---

## Model Training & Evaluation

### **Models Implemented:**
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **Multilayer Perceptrons (MLP)**
4. **XGBoost Classifier**
5. **Support Vector Machine (SVM)**

### **Performance Metrics:**
- **Accuracy**: Measures overall correctness.
- **Precision**: Determines how many predicted phishing URLs were actually phishing.
- **Confusion Matrix**: Visual representation of model performance.
- **Feature Importance Analysis**: Evaluates the impact of different features on predictions.
- **Cross-Validation**: Ensures model performance is stable across different data splits.

### **Results:**
- The best-performing model was **XGBoost** with **96% test accuracy and precision**.
- Decision Trees and Random Forests also provided strong results, with interpretable decision paths.
- **Feature importance analysis** highlighted:
  - `URL Entropy`
  - `Domain Age`
  - `TinyURL` usage
  - `HTTPS in Domain`
  - `Redirection`
- **Real-time URL Detection**: Predicts phishing URLs in under **2 seconds** due to optimized feature extraction.

---

## Model Persistence (Storage & Loading)
To improve efficiency, trained models are now **saved and reloaded** instead of requiring retraining on every launch.

### **How Model Storage Works**
- After training, models are stored in a **pickle (`.pkl`) file**.
- Instead of retraining, users can **load pretrained models** instantly.
- This reduces startup time and allows quick URL classification.
- **Automated Model Loading**: The Streamlit app automatically loads models if available.

### **How to Train & Save Models (One-Time Setup)**
```python
import pickle

# Save trained models to a file
trained_models = {
    'Decision Tree': decision_tree_model,
    'Random Forest': random_forest_model,
    'MLP': mlp_model,
    'XGBoost': xgb_model,
    'SVM': svm_model
}

with open('trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
```

### **How to Load Models in Streamlit**
```python
import pickle

# Load pretrained models
with open('trained_models.pkl', 'rb') as f:
    loaded_models = pickle.load(f)
```
- The Streamlit app includes a **"Load Pretrained Models"** button to load models automatically.

---

## Streamlit Dashboard
The **Streamlit dashboard** provides an interactive interface for data exploration and real-time phishing detection.

### **Dashboard Features**
1. **Data Overview**: Displays dataset statistics and feature explanations.
2. **Exploratory Data Analysis (EDA)**: Allows users to visualize distributions and correlations.
3. **Train & Evaluate Models**: Enables training and validation of multiple models.
4. **Model Comparison**: Summarizes and ranks models based on performance.
5. **Live URL Checker**: Accepts URLs and predicts their legitimacy using the trained model.
6. **Batch URL Detection**: Allows users to input multiple URLs and receive batch predictions.

### **How to Use the Live URL Checker**
1. Go to the **"Phishing URL Detector"** tab.
2. Enter a URL.
3. Click **"Check URL"**.
4. The model classifies the URL as **Phishing** or **Legit**.

---

## Installation & Setup
### **Prerequisites:**
- Python 3.8+
- Pip & Virtual Environment
- Streamlit, Scikit-learn, XGBoost, Pandas, NumPy, Plotly
- **Requests, Threading** (for optimized URL processing)

### **Installation Steps:**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourrepo/phishing-detector.git
   cd phishing-detector
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Load Pretrained Models**
   - Download the `trained_models.pkl` file (or train models manually).
   - Place it in the project directory.

5. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## Future Improvements
- **Enhance Feature Engineering:** Incorporate content-based features by scraping webpage text.
- **Improve WHOIS Reliability:** Use a dedicated WHOIS API for real-time domain age retrieval.
- **Deep Learning Models:** Experiment with LSTMs or CNNs for URL pattern analysis.
- **Expand Dataset:** Integrate more real-world phishing URL sources.
- **Automated Model Retraining:** Schedule periodic updates for model retraining with fresh data.

---

## License
This project is licensed under the MIT License.

---

## References
- PhishTank (https://www.phishtank.com/)
- University of New Brunswick (https://www.unb.ca/cic/datasets/url-2016.html)
- Scikit-learn documentation (https://scikit-learn.org/)
- XGBoost documentation (https://xgboost.readthedocs.io/)

---