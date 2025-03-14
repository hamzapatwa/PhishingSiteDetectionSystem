import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt  # (Optional if you want local visualizations)
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
from requests.exceptions import SSLError, Timeout, RequestException
import whoisdomain
from datetime import datetime
import math
import subprocess
import whois

###############################################################################
#                           1. HELPER FUNCTIONS
###############################################################################

@st.cache_data
def load_data():
    """
    Load the CSV data containing extracted URL features (with new columns).
    The file should be named 'urldata.csv' and located in the same directory.
    """
    df = pd.read_csv('urldata.csv')  # Adjust path if needed
    return df

@st.cache_resource
def load_pretrained_models(file_path="trained_models.pkl"):
    """Load models from disk and store in session state."""
    if "model_results" not in st.session_state:
        with open(file_path, 'rb') as f:
            st.session_state["model_results"] = pickle.load(f)
    return st.session_state["model_results"]


def plot_conf_matrix(y_true, y_pred, labels=("Legit", "Phish"), title="Confusion Matrix"):
    """
    Generate an interactive Plotly confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title,
        aspect="auto"
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return fig


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train multiple machine learning models and compute their performance metrics.
    Models trained:
      - Decision Tree
      - Random Forest
      - Multilayer Perceptrons (MLP)
      - XGBoost

    Returns:
    - Dictionary containing each model's object, predictions, accuracy,
      precision, and the confusion matrix plot (Plotly figure).
    """
    model_results = {}

    def store_model(name, model, y_pred_train, y_pred_test):
        """
        Helper function to compute metrics and store results for a given model.
        """
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        prec_train = precision_score(y_train, y_pred_train)
        prec_test = precision_score(y_test, y_pred_test)
        cm_fig = plot_conf_matrix(y_test, y_pred_test, title=f"{name} - Confusion Matrix")
        return {
            "model_name": name,
            "model_object": model,
            "acc_train": acc_train,
            "acc_test": acc_test,
            "prec_train": prec_train,
            "prec_test": prec_test,
            "conf_matrix_fig": cm_fig
        }

    # 1) Decision Tree
    tree = DecisionTreeClassifier(max_depth=20, random_state=42)
    tree.fit(X_train, y_train)
    model_results["Decision Tree"] = store_model(
        "Decision Tree",
        tree,
        tree.predict(X_train),
        tree.predict(X_test)
    )

    # 2) Random Forest
    forest = RandomForestClassifier(max_depth=40, random_state=42)
    forest.fit(X_train, y_train)
    model_results["Random Forest"] = store_model(
        "Random Forest",
        forest,
        forest.predict(X_train),
        forest.predict(X_test)
    )

    # 3) MLP
    mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(200, 200, 200), random_state=42)
    mlp.fit(X_train, y_train)
    model_results["Multilayer Perceptrons"] = store_model(
        "Multilayer Perceptrons",
        mlp,
        mlp.predict(X_train),
        mlp.predict(X_test)
    )

    # 4) XGBoost
    xgb = XGBClassifier(learning_rate=0.4, max_depth=42, random_state=42)
    xgb.fit(X_train, y_train)
    model_results["XGBoost"] = store_model(
        "XGBoost",
        xgb,
        xgb.predict(X_train),
        xgb.predict(X_test)
    )

    return model_results


def get_tree_graph(tree_model, feature_names):
    """
    Generate a Graphviz source for the given Decision Tree model.
    """
    dot_data = export_graphviz(
        tree_model,
        out_file=None,
        feature_names=feature_names,
        class_names=["Legit", "Phish"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    return graphviz.Source(dot_data)


###############################################################################
# 2. ENHANCED PHISHING URL DETECTOR FUNCTIONS
###############################################################################

def calculate_entropy(s):
    """Calculate the Shannon entropy of a string."""
    if not s:
        return 0
    probs = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum(p * math.log(p, 2) for p in probs)

def count_subdomains(domain):
    tokens = domain.split('.')
    return max(0, len(tokens) - 2)

def count_digits(url):
    return sum(c.isdigit() for c in url)

def count_special_chars(url):
    # Count characters that are not alphanumeric and not common punctuation
    return sum(1 for c in url if not c.isalnum() and c not in ".:/?-_")

def uppercase_ratio(url):
    if len(url) == 0:
        return 0
    return sum(1 for c in url if c.isupper()) / len(url)

def get_domain_age(domain):
    try:
        import whois  # Ensure you're importing the correct whois library
        w = whois.whois(domain)
        creation_date = w.creation_date
        # Sometimes creation_date is a list, so take the first element
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date is None:
            print("No creation date found in WHOIS data")
            return -1
        age_days = (datetime.now() - creation_date).days
        print(f"Domain age: {age_days} days (created on {creation_date})")
        return age_days
    except Exception as e:
        print(f"Error: {e}")
        return -1

def get_http_response(url):
    try:
        resp = requests.get(url, timeout=5)
        return resp
    except RequestException:
        return ""

def iframe_check(response):
    if not response or response == "":
        return 1
    return 0 if re.search(r"<iframe|<frameBorder", response.text, re.IGNORECASE) else 1

def mouse_over_check(response):
    if not response or response == "":
        return 1
    return 1 if re.search(r"<script.*onmouseover.*</script>", response.text, re.IGNORECASE | re.DOTALL) else 0

def right_click_check(response):
    if not response or response == "":
        return 1
    return 0 if re.search(r"event\.button\s*==\s*2", response.text) else 1

def forwarding_check(response):
    if not response or response == "":
        return 1
    return 1 if len(response.history) > 2 else 0

def is_ip_address(url):
    # Simple utility to check if the netloc is an IP
    try:
        ipaddress.ip_address(url)
        return True
    except:
        return False

def convert_features_to_numeric(feat_list):
    """
    Convert any string placeholders (like domain) to 0 or drop them if your final
    model doesn't expect them. If your trained model expects 'Domain' to be dropped,
    remove it. Otherwise, handle it.
    """
    if isinstance(feat_list[0], str):
        feat_list = feat_list[1:]  # Remove the first item, which is domain
    return [float(val) if not isinstance(val, str) else 0.0 for val in feat_list]

def extract_features(url: str) -> list:
    """
    Extracts an expanded set of features from a single URL, including:
      1) Address-bar features (IP presence, '@', length, depth, redirection, etc.)
      2) Lexical / Statistical features (entropy, digit counts, subdomains, etc.)
      3) WHOIS domain age
      4) HTML/JS-based features from the live HTTP response
    Returns a list of numeric feature values that match the final dataset’s column order.
    """
    # 1. Address bar-based features
    domain = urlparse(url).netloc
    if re.match(r"^www\.", domain):
        domain = domain.replace("www.", "")

    ip_feat = 1 if is_ip_address(url) else 0
    at_feat = 1 if "@" in url else 0
    length_feat = 1 if len(url) >= 54 else 0
    depth_feat = sum(1 for part in urlparse(url).path.split('/') if part)
    redirect_feat = 1 if url.rfind('//') > 7 else 0
    https_in_domain_feat = 1 if 'https' in domain else 0

    # Shortening Services
    short_regex = (
        r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
        r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
        r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
        r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
        r"tr\.im|link\.zip\.net"
    )
    tiny_feat = 1 if re.search(short_regex, url) else 0
    prefix_feat = 1 if '-' in domain else 0

    # 2. Lexical / Statistical Features
    url_entropy = calculate_entropy(url)
    domain_entropy = calculate_entropy(domain)
    subdomain_ct = count_subdomains(domain)
    digit_ct = count_digits(url)
    special_char_ct = count_special_chars(url)
    uppercase_rat = uppercase_ratio(url)
    domain_age = get_domain_age(domain)

    # 3. HTML-based features
    response = get_http_response(url)
    iframe_feat = iframe_check(response)
    mouse_feat = mouse_over_check(response)
    right_click_feat = right_click_check(response)
    forward_feat = forwarding_check(response)

    # 4. Build the final feature list in the same order as your improved dataset
    features_list = [
        domain,
        ip_feat,
        at_feat,
        length_feat,
        depth_feat,
        redirect_feat,
        https_in_domain_feat,
        tiny_feat,
        prefix_feat,
        url_entropy,
        domain_entropy,
        subdomain_ct,
        digit_ct,
        special_char_ct,
        uppercase_rat,
        domain_age,
        iframe_feat,
        mouse_feat,
        right_click_feat,
        forward_feat,
    ]

    return convert_features_to_numeric(features_list)


###############################################################################
#                           STREAMLIT APP LAYOUT
###############################################################################
st.set_page_config(page_title="Phishing Detection App", layout="wide")
st.title("Phishing URL Detection — Full Analysis (Improved)")
st.markdown(
    """
    **New Update:**  
    We've now integrated improved WHOIS lookups and batch processing for feature extraction.  
    This means we can now process virtually unlimited URLs (previously capped at ~30,000) without running out of memory.
    """
)

# Load the data
df = load_data()

# Tabs
tab_overview, tab_eda, tab_modeling, tab_compare, tab_url_checker = st.tabs(
    [
        "Data Overview",
        "EDA",
        "Train & Evaluate Models",
        "Compare All Models",
        "Phishing URL Detector"
    ]
)

###############################################################################
# 3.A  Data Overview Tab
###############################################################################
with tab_overview:
    st.subheader("1. Data Overview")
    st.markdown(
        """
        The dataset **urldata.csv** contains URL-derived features. Each row represents one URL
        and whether it's labeled as *Phish* (1) or *Legit* (0).

        **Updates**:
        - Integrated WHOIS lookups and batch processing to efficiently handle large datasets.
        - New lexical/statistical features added (e.g., entropy, subdomain counts, uppercase ratio).
        - Memory management improvements allow us to scale beyond ~30,000 URLs.

        **Key Features**:
        - `Having_IP`: Whether the domain portion is purely an IP address.
        - `Have_At`: Checks '@' symbol usage.
        - `URL_Length`: ≥54 chars is suspicious.
        - `URL_Depth`: Count of '/' segments.
        - `Redirection`: Looks for '//' beyond protocol.
        - `HTTPS_in_Domain`: 'https' found in domain name.
        - `TinyURL`: Known link-shortener detection.
        - `Prefix/Suffix`: '-' in domain.
        - `Iframe`, `Mouse_Over`, `Right_Click`, `Web_Forwards`: Suspicious HTML/JS behaviors.
        - `Lexical Features`: Entropy, uppercase ratio, domain age, subdomains, digit count, etc.

        **Shape of data**:
        """
    )
    st.write(df.shape)
    st.dataframe(df.head())

    if "Domain" in df.columns:
        st.write("**Note**: The 'Domain' column is dropped in modeling (non-numeric).")
    else:
        st.write("**Domain** column not present or already removed.")

    st.write("### Missing values per column:")
    st.write(df.isnull().sum())

    st.info(
        """
        **Goal**:  
        1) Explore data (EDA).  
        2) Train classification models (Decision Tree, Random Forest, MLP, XGBoost).  
        3) Compare model performance & interpret feature importances.  
        4) Provide a live URL checker using the newly trained or loaded models.
        """
    )

###############################################################################
# 3.B  EDA Tab
###############################################################################
with tab_eda:
    st.subheader("2. Exploratory Data Analysis (EDA)")

    # If 'Domain' column exists, drop for numeric analysis
    if "Domain" in df.columns:
        data_eda = df.drop("Domain", axis=1).copy()
    else:
        data_eda = df.copy()

    data_eda = data_eda.sample(frac=1, random_state=42).reset_index(drop=True)

    st.markdown("### Distribution of a Numeric Column")
    numeric_cols = [
        col for col in data_eda.columns
        if data_eda[col].dtype in [np.int64, np.float64] and col != 'Label'
    ]

    if numeric_cols:
        chosen_col = st.selectbox("Select a numeric column:", numeric_cols)
        fig_hist = px.histogram(data_eda, x=chosen_col, title=f"Distribution of {chosen_col}")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.write("No numeric columns to visualize distributions.")

    st.markdown("### Correlation Heatmap")
    corr_data = data_eda.drop("Label", axis=1).select_dtypes(include=[np.number])
    if len(corr_data.columns) > 1:
        fig_corr = px.imshow(
            corr_data.corr(),
            text_auto=".2f",
            title="Correlation Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.write("Not enough numeric columns to display a correlation heatmap.")

###############################################################################
# 3.C  Train & Evaluate Models Tab
###############################################################################
with tab_modeling:
    st.subheader("3. Train & Evaluate Models")

    if "model_results" in st.session_state:
        st.success("✅ Pretrained models are already loaded! No need to reload.")
    else:
        st.warning("No models found. Load or train models below.")

    # Prep data
    if "Domain" in df.columns:
        data_model = df.drop(["Domain"], axis=1).copy()
    else:
        data_model = df.copy()
    data_model = data_model.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data_model.drop("Label", axis=1)
    y = data_model["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    st.write("**Training Set Shape**:", X_train.shape)
    st.write("**Test Set Shape**:", X_test.shape)

    # --- Option to Load Pretrained or Train from Scratch ---
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Load Pretrained Models from Disk"):
            loaded_models_dict = load_pretrained_models("trained_models.pkl")
            model_results = {}
            for m_name, m_obj in loaded_models_dict.items():
                y_pred_train = m_obj.predict(X_train)
                y_pred_test = m_obj.predict(X_test)

                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)
                prec_train = precision_score(y_train, y_pred_train)
                prec_test = precision_score(y_test, y_pred_test)

                cm_fig = plot_conf_matrix(y_test, y_pred_test, title=f"{m_name} - Confusion Matrix")
                model_results[m_name] = {
                    "model_name": m_name,
                    "model_object": m_obj,
                    "acc_train": acc_train,
                    "acc_test": acc_test,
                    "prec_train": prec_train,
                    "prec_test": prec_test,
                    "conf_matrix_fig": cm_fig
                }

            st.session_state["model_results"] = model_results
            st.success("Loaded pretrained models from 'trained_models.pkl' and evaluated them.")

    with col2:
        if st.button("Train All Models"):
            st.write("Training models... please wait.")
            model_results = train_all_models(X_train, y_train, X_test, y_test)
            st.session_state["model_results"] = model_results
            st.success("Training complete!")

    # Show results if available
    if "model_results" in st.session_state:
        st.write("### Model Results")
        model_results = st.session_state["model_results"]

        for name, res in model_results.items():
            with st.expander(f"{name} Results", expanded=False):
                st.write(f"**Accuracy (Train):** {res['acc_train']:.3f}")
                st.write(f"**Accuracy (Test):** {res['acc_test']:.3f}")
                st.write(f"**Precision (Train):** {res['prec_train']:.3f}")
                st.write(f"**Precision (Test):** {res['prec_test']:.3f}")
                st.plotly_chart(res["conf_matrix_fig"], use_container_width=True)

    else:
        st.info("No models loaded or trained yet. Use one of the above buttons.")

###############################################################################
# 3.D  Compare All Models Tab
###############################################################################
with tab_compare:
    st.subheader("4. Compare All Models")
    if "model_results" not in st.session_state:
        st.warning("No trained/loaded models. Go to 'Train & Evaluate Models' to proceed.")
    else:
        model_results = st.session_state["model_results"]
        summary_data = []
        for v in model_results.values():
            summary_data.append({
                "ML Model": v["model_name"],
                "Train Accuracy": round(v["acc_train"], 3),
                "Test Accuracy": round(v["acc_test"], 3),
                "Train Precision": round(v["prec_train"], 3),
                "Test Precision": round(v["prec_test"], 3)
            })
        summary_df = pd.DataFrame(summary_data).sort_values(
            by=["Test Accuracy", "Test Precision", "Train Accuracy", "Train Precision"],
            ascending=False
        ).reset_index(drop=True)

        st.dataframe(summary_df)
        st.markdown("**Top model** is listed first based on test accuracy & precision.")

        st.write("### Feature Importances")
        st.write("For models that support feature importances (Random Forest, XGBoost, Decision Tree):")
        model_choice = st.selectbox("Pick a model:", ["Random Forest", "XGBoost", "Decision Tree"])
        if model_choice in model_results:
            model_obj = model_results[model_choice]["model_object"]
            importances = getattr(model_obj, "feature_importances_", None)
            if importances is not None:
                feat_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                fig_fi = px.bar(
                    x=feat_series.values,
                    y=feat_series.index,
                    orientation='h',
                    title=f"{model_choice} Feature Importances"
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importances are not available for this model.")
        else:
            st.info("Please train/load the selected model first.")

###############################################################################
# 3.E  Phishing URL Detector Tab (With Model Selection)
###############################################################################
with tab_url_checker:
    st.subheader("5. Live Phishing URL Detector (New Features)")
    st.write("Enter a URL to check if it's phishing or not, using one of the loaded/trained models.")

    if "model_results" not in st.session_state:
        st.warning("No models available. Please train or load models in 'Train & Evaluate Models' tab.")
    else:
        model_names = list(st.session_state["model_results"].keys())
        selected_model = st.selectbox("Select Model for Prediction:", model_names)
        url_input = st.text_input("Enter URL to check:", "")

        if st.button("Check URL"):
            if not url_input.strip():
                st.warning("Please enter a valid URL.")
            else:
                st.write("Extracting advanced features (domain age, lexical stats, etc.)...")
                # Get the selected model object
                model_obj = st.session_state["model_results"][selected_model]["model_object"]

                # Extract features from the URL
                features_vector = extract_features(url_input)

                # Define feature columns: drop 'Domain' and 'Label' from df for training
                if "Domain" in df.columns:
                    feature_cols = list(df.drop(["Domain", "Label"], axis=1).columns)
                else:
                    feature_cols = list(df.drop("Label", axis=1).columns)

                # Convert the extracted features to a DataFrame using the defined columns
                features_df = pd.DataFrame([features_vector], columns=feature_cols)

                # Make prediction using the model
                prediction = model_obj.predict(features_df)

                # Interpret prediction (assuming 1=Phish, 0=Legit)
                if prediction[0] == 1:
                    st.error(f"🚨 **Phishing Alert!** `{selected_model}` classifies this URL as phishing.")
                else:
                    st.success(f"✅ **Safe URL!** `{selected_model}` says this seems legit.")

        st.markdown(
            """
            **How It Works**:
            - We extract an expanded feature set (domain age, lexical distribution, subdomain counts, etc.).
            - The selected trained model then classifies it as either *Phishing* or *Legit*.
            """
        )