# **Phishing URL Detection System**

## **Overview**
This project detects phishing websites using machine learning models trained on extracted URL features. **Phishing websites mimic legitimate ones** to trick users into entering sensitive information. Our system applies **advanced feature extraction, model optimization, and real-time URL classification** via a **Streamlit dashboard**.

## **What's New?**
🚀 **Major Updates & Optimizations**:
- **Achieved >99% Test Accuracy & Precision** 🎯 through **model finetuning**.
- **WHOIS Domain Age Lookup Optimized** ⏳: Parallel WHOIS queries speed up processing.
- **Batch Processing for Large Datasets** 📊: We now process **unlimited URLs** (previously ~30K limit).
- **Improved URL Feature Extraction** 🔍: Now **20+ features** covering lexical, statistical, and HTML-based signals.
- **Pretrained Models for Instant Predictions** 🚀: No need to retrain every time!
- **Live URL Checker** 🌐: Paste a URL and get a phishing verdict in **<2 seconds**.

---

## **Features & Capabilities**
✔ **Handles Large Datasets**: **100,000 URLs** (50K phishing, 50K legitimate).  
✔ **Advanced Feature Engineering**: URL-based, lexical, statistical, WHOIS, and JavaScript-based indicators.  
✔ **Parallel Processing**: **Multithreading** for HTTP/WHOIS requests speeds up processing.  
✔ **Optimized Batch Processing**: Efficient **disk storage frees up memory** for large datasets.  
✔ **Machine Learning Models**: Uses **Decision Trees, Random Forests, XGBoost, MLP**.  
✔ **Streamlit Dashboard**: User-friendly **model evaluation & real-time phishing detection**.  
✔ **Pretrained Model Storage**: Saves models in `.pkl` format for **fast loading without retraining**.  

---

## **Dataset**
We trained on **100,000 URLs**, sourced from:
- **PhishTank** 🛑: Verified phishing URLs.
- **University of New Brunswick** 📚: Dataset with phishing & legitimate URLs.
- **Additional Real-World Data** 🌍: Manually collected and verified sources.

---

## **Feature Engineering**
We extract **20+ features** to identify phishing URLs.

### **🔗 Address Bar-Based Features**
- **Having_IP**: Whether the domain is an IP (common in phishing).
- **Have_At**: Presence of '@' (phishers use this to trick users).
- **URL_Length**: Long URLs (>54 characters) are more likely phishing.
- **URL_Depth**: More `/` indicates a complex path, sometimes phishing.
- **Redirection**: Checks double `//` after protocol (`http://site.com//evil`).
- **HTTPS in Domain**: Phishers may **add "https"** in the domain name (not secure).
- **TinyURL Usage**: Detects shortened links (bit.ly, goo.gl).
- **Prefix/Suffix**: `-` in domain (`secure-paypal.com` is likely phishing).

### **📊 Lexical & Statistical Features**
- **URL Entropy**: Higher entropy suggests randomness (phishing links are gibberish).
- **Domain Entropy**: Measures complexity of domain structure.
- **Subdomain Count**: More subdomains (`login.bank.secure.com`) suggest phishing.
- **Digit Count**: Excessive digits in the URL (e.g., `free-prize123.com`) raise suspicion.
- **Special Character Count**: Symbols like `_`, `-`, `%`, `=`, common in phishing URLs.
- **Uppercase Ratio**: Random uppercase letters in a domain are suspicious.
- **Domain Age (WHOIS)**: Old domains are **trustworthy**; phishing sites are often new.

### **💻 HTML & JavaScript Features**
- **iFrame Detection**: Hidden `iframe` tags are used for phishing.
- **Mouse Over JavaScript**: JS altering cursor behavior is suspicious.
- **Right Click Disabled**: Blocks user actions to hide phishing.
- **Web Forwarding**: Too many redirects indicate phishing.

---

## **🚀 Model Training & Evaluation**
We finetuned our models and **now achieve >99% test accuracy and precision**!

### **📊 Models Implemented**
1. **Decision Tree Classifier 🌳**
2. **Random Forest Classifier 🌲**
3. **Multilayer Perceptrons (MLP) 🧠**
4. **XGBoost Classifier ⚡**

### **📈 Performance Metrics**
- ✅ **Accuracy**: Correct classification rate.
- 🎯 **Precision**: % of predicted phishing URLs that were actually phishing.
- 🟩 **Confusion Matrix**: Visual error analysis.
- 📌 **Feature Importance Analysis**: Identifies key phishing indicators.
- 🔍 **Cross-Validation**: Ensures consistency across different data splits.

### **🏆 Results:**
- **XGBoost** is the best model with **99.2% accuracy & precision**.
- **Random Forest & Decision Trees** also performed well with high interpretability.
- **Feature importance highlights**:
  - `URL Entropy`
  - `Domain Age`
  - `TinyURL`
  - `Redirection`
  - `HTTPS in Domain`
- **Real-time detection speed**: **<2 seconds** per URL.

---

## **💾 Model Persistence (Save & Load)**
To improve efficiency, **models are stored & reloaded** instead of retraining every launch.

### **📌 How to Save Models**
```python
import pickle

# Save trained models to a file
trained_models = {
    'Decision Tree': decision_tree_model,
    'Random Forest': random_forest_model,
    'MLP': mlp_model,
    'XGBoost': xgb_model
}

with open('trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)
```

### **📌 How to Load Models in Streamlit**
```python
import pickle

# Load pretrained models
with open('trained_models.pkl', 'rb') as f:
    loaded_models = pickle.load(f)
```
- **The Streamlit dashboard automatically loads models** if available.

---

## **🚀 Future Improvements**
🔹 **Enhance Feature Engineering**: Extract webpage content-based signals.  
🔹 **Improve WHOIS API Reliability**: Use a dedicated API for domain lookups.  
🔹 **Deep Learning**: Experiment with **LSTMs** or **CNNs** for URL analysis.  
🔹 **Automated Retraining**: Periodically refresh model with new data.  

---

## **📜 License**
This project is licensed under the **MIT License**.

---

## **📚 References**
- PhishTank (https://www.phishtank.com/)
- University of New Brunswick (https://www.unb.ca/cic/datasets/url-2016.html)
- Scikit-learn (https://scikit-learn.org/)
- XGBoost (https://xgboost.readthedocs.io/)

