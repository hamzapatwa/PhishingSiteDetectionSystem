# utils.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import graphviz
import ipaddress
import re
import requests
import pickle
from urllib.parse import urlparse
from requests.exceptions import RequestException
from datetime import datetime
import math
import time  # For simulating progress

# --- Data and Model Loading ---

@st.cache_data
def load_data(file_path='urldata.csv'):
    """Loads the dataset."""
    try:
        df = pd.read_csv(file_path)
        # Simple check for expected columns - adjust if needed
        if 'Label' not in df.columns:
             st.error(f"Critical Error: 'Label' column not found in {file_path}. Cannot proceed.")
             st.stop()
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset file not found at `{file_path}`. Please ensure it's available.")
        st.stop() # Stop execution if data isn't loaded
    except Exception as e:
        st.error(f"An error occurred loading data from `{file_path}`: {e}")
        st.stop()


@st.cache_resource
def load_pretrained_models(file_path="trained_models.pkl"):
    """Loads pretrained models from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            pretrained_data = pickle.load(f)
        # Store in session state if not already there
        if "pretrained_models_loaded" not in st.session_state:
             st.session_state["pretrained_models_data"] = pretrained_data
             st.session_state["pretrained_models_loaded"] = True
        return st.session_state["pretrained_models_data"]
    except FileNotFoundError:
        st.error(f"Error: Pretrained models file not found at `{file_path}`.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading pretrained models from `{file_path}`: {e}")
        return None


# --- Plotting Functions ---

def plot_conf_matrix(y_true, y_pred, labels=("Legit (0)", "Phish (1)"), title="Confusion Matrix"):
    """Generates a Plotly confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
        labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
        aspect="auto"
    )
    fig.update_xaxes(side="bottom")
    fig.update_layout(title_x=0.5) # Center title
    return fig

def plot_feature_importance(model, feature_names, model_name):
    """Generates a Plotly bar chart for feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        fig = px.bar(
            x=sorted_importances,
            y=sorted_features,
            orientation='h',
            title=f"{model_name} - Feature Importances",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            text=np.round(sorted_importances, 3) # Show values on bars
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, title_x=0.5)
        return fig
    else:
        st.info(f"Feature importances are not available for the {model_name} model.")
        return None

def plot_model_comparison(summary_df):
    """Generates grouped bar charts for comparing model metrics."""
    df_melt = summary_df.melt(id_vars="ML Model",
                              value_vars=["Train Accuracy", "Test Accuracy", "Train Precision", "Test Precision"],
                              var_name="Metric", value_name="Score")

    fig = px.bar(df_melt, x="ML Model", y="Score", color="Metric",
                 barmode="group", title="Model Performance Comparison",
                 labels={'Score': 'Metric Score'},
                 text_auto=".3f") # Format text on bars
    fig.update_layout(title_x=0.5)
    fig.update_traces(textangle=0, textposition="outside")
    return fig

# --- Model Training and Evaluation ---

def store_model_result(name, model, y_pred_train, y_pred_test, y_train, y_test):
    """Calculates metrics and stores model results."""
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_train = precision_score(y_train, y_pred_train, zero_division=0) # Handle zero division
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    cm_fig = plot_conf_matrix(y_test, y_pred_test, title=f"{name} - Test Set Confusion Matrix")

    return {
        "model_name": name,
        "model_object": model,
        "acc_train": acc_train,
        "acc_test": acc_test,
        "prec_train": prec_train,
        "prec_test": prec_test,
        "conf_matrix_fig": cm_fig
    }

# Individual model training functions (with progress simulation)
def train_decision_tree_model(X_train, y_train, X_test, y_test, max_depth, progress_bar):
    progress = 0
    progress_bar.progress(progress, text="Initializing Decision Tree...")
    time.sleep(0.2)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    progress_bar.progress(25, text="Fitting Decision Tree...")
    tree.fit(X_train, y_train)
    progress_bar.progress(75, text="Predicting and Evaluating...")
    time.sleep(0.3)
    result = store_model_result(
        "Decision Tree", tree, tree.predict(X_train), tree.predict(X_test), y_train, y_test
    )
    progress_bar.progress(100, text="Decision Tree Training Complete!")
    time.sleep(0.5)
    return result

def train_random_forest_model(X_train, y_train, X_test, y_test, max_depth, n_estimators, progress_bar):
    progress = 0
    progress_bar.progress(progress, text="Initializing Random Forest...")
    time.sleep(0.2)
    forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42, n_jobs=-1) # Use all cores
    progress_bar.progress(10, text=f"Fitting Random Forest ({n_estimators} trees)...")
    forest.fit(X_train, y_train)
    progress_bar.progress(80, text="Predicting and Evaluating...")
    time.sleep(0.3)
    result = store_model_result(
        "Random Forest", forest, forest.predict(X_train), forest.predict(X_test), y_train, y_test
    )
    progress_bar.progress(100, text="Random Forest Training Complete!")
    time.sleep(0.5)
    return result

def train_mlp_model(X_train, y_train, X_test, y_test, alpha, hidden_layer_sizes, progress_bar):
    progress = 0
    progress_bar.progress(progress, text="Initializing MLP...")
    time.sleep(0.2)
    mlp = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=42, max_iter=500, early_stopping=True) # Added sensible defaults
    progress_bar.progress(15, text="Fitting MLP (this might take a moment)...")
    mlp.fit(X_train, y_train)
    progress_bar.progress(85, text="Predicting and Evaluating...")
    time.sleep(0.3)
    result = store_model_result(
        "MLP", mlp, mlp.predict(X_train), mlp.predict(X_test), y_train, y_test
    )
    progress_bar.progress(100, text="MLP Training Complete!")
    time.sleep(0.5)
    return result

def train_xgboost_model(X_train, y_train, X_test, y_test, learning_rate, max_depth, n_estimators, progress_bar):
    progress = 0
    progress_bar.progress(progress, text="Initializing XGBoost...")
    time.sleep(0.2)
    xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, random_state=42, use_label_encoder=False, eval_metric='logloss') # Common practice
    progress_bar.progress(10, text=f"Fitting XGBoost ({n_estimators} rounds)...")
    xgb.fit(X_train, y_train)
    progress_bar.progress(80, text="Predicting and Evaluating...")
    time.sleep(0.3)
    result = store_model_result(
        "XGBoost", xgb, xgb.predict(X_train), xgb.predict(X_test), y_train, y_test
    )
    progress_bar.progress(100, text="XGBoost Training Complete!")
    time.sleep(0.5)
    return result

# --- Feature Extraction for Live Detector ---

# WHOIS requires installation: pip install python-whois
# It can be slow and sometimes unreliable depending on registry throttling or domain privacy.
# Provide a fallback mechanism.
_ENABLE_WHOIS = True # Set to False to disable potentially slow WHOIS lookups
try:
    import whois
except ImportError:
    st.sidebar.warning("`python-whois` not installed. Domain age feature will be disabled. Install with `pip install python-whois`", icon="⚠️")
    _ENABLE_WHOIS = False


def get_domain_age(domain):
    """Attempts to get domain age in days. Returns -1 on failure or if disabled."""
    if not _ENABLE_WHOIS:
        return -1
    try:
        # Ensure domain is just the domain name, not subdomain
        parts = domain.split('.')
        if len(parts) > 2:
            domain = '.'.join(parts[-2:]) # Get the main domain like example.com from sub.example.com

        w = whois.query(domain) # Use query instead of whois for potentially better results
        if w and w.creation_date:
            creation_date = w.creation_date
            if isinstance(creation_date, list): # Sometimes returns a list
                creation_date = creation_date[0]
            if isinstance(creation_date, datetime):
                 age_days = (datetime.now() - creation_date).days
                 return age_days if age_days >= 0 else -1 # Ensure non-negative age
        return -1 # Indicate failure or no data
    except Exception as e:
        # st.sidebar.warning(f"WHOIS lookup failed for {domain}: {e}", icon="🕸️") # Optional warning
        return -1 # Indicate failure

def calculate_entropy(s):
    """Calculates Shannon entropy of a string."""
    if not s: return 0
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0.) + 1
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def count_subdomains(domain):
    """Counts subdomains (e.g., mail.example.com -> 1 subdomain)."""
    # Remove www. prefix before counting
    if domain.startswith('www.'):
        domain = domain[4:]
    parts = domain.split('.')
    # Basic TLDs like .com, .org are not subdomains.
    # Consider co.uk, .com.au etc. (2 parts) - adjust if needed for more complex TLDs
    if len(parts) <= 2: # e.g., example.com
        return 0
    elif len(parts) == 3 and len(parts[-2]) <= 3: # e.g., example.co.uk
         return 0
    else: # e.g., mail.example.com or mail.example.co.uk
        # Assuming the last 2 (or 3 for co.uk type) parts are the main domain + TLD
        tld_parts = 2
        if len(parts) > 2 and len(parts[-2]) <= 3 and len(parts[-1]) <=3: # Heuristic for .co.uk etc.
             tld_parts = 3
        return len(parts) - tld_parts


def extract_features(url: str, expected_feature_names: list) -> pd.DataFrame | None:
    """Extracts features from a single URL based on the expected training columns."""
    features = {}
    try:
        # Ensure URL has a scheme
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url # Default to http if scheme missing

        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        full_url_len = len(url)

        # Clean domain (remove port if present)
        if ':' in domain:
            domain = domain.split(':')[0]

        # --- Feature Extraction Logic (aligned with common phishing features) ---
        features['Having_IP'] = 1 if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", domain) else 0
        features['Have_At'] = 1 if '@' in url else 0
        # Length thresholds can vary; 54 is common, but adjust if needed based on original data analysis
        features['URL_Length'] = 1 if full_url_len >= 54 else (0 if full_url_len < 54 else -1) # Use -1, 0, 1 format if needed
        features['URL_Depth'] = path.count('/')
        features['Redirection'] = 1 if '//' in path else 0 # Simple check for // in path
        features['https_Domain'] = 1 if 'https' in domain else 0 # Check for 'https' string in domain part
        features['TinyURL'] = 1 if domain in ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'ow.ly'] else 0 # Basic shorteners
        features['Prefix/Suffix'] = 1 if '-' in domain else 0

        # Advanced / Content-Based (Optional - require network requests, can be slow/fail)
        # These often correspond to columns like 'Iframe', 'Mouse_Over', 'Right_Click', 'Web_Forwards'
        # For a live detector, these can be problematic. We'll use placeholder values (-1 for unavailable).
        features['iFrame'] = -1
        features['Mouse_Over'] = -1
        features['Right_Click'] = -1
        features['Web_Forwards'] = -1

        # WHOIS Feature
        features['Domain_Age'] = get_domain_age(domain) if _ENABLE_WHOIS else -1 # Use -1 if unavailable/error

        # Example: Add other common features if they were in the original dataset
        features['Domain_registration_length'] = -1 # Often requires WHOIS, placeholder
        features['Google_Index'] = -1 # Requires search engine query, placeholder
        features['Page_Rank'] = -1 # Deprecated metric, placeholder

        # --- Create DataFrame with correct columns ---
        # Ensure all expected columns are present, filling missing ones with a default (e.g., 0 or -1)
        final_features = {}
        for col_name in expected_feature_names:
             final_features[col_name] = features.get(col_name, 0) # Default to 0 if not extracted

        feature_df = pd.DataFrame([final_features], columns=expected_feature_names)
        return feature_df

    except Exception as e:
        st.error(f"Error extracting features for '{url}': {e}")
        return None

# --- Other Helpers ---
def safe_rerun():
    """Reruns the script safely, handling potential attribute errors."""
    try:
        st.rerun()
    except Exception: # Catch broad exception if rerun fails for any reason
         st.experimental_rerun() # Fallback for older versions