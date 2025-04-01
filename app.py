import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
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
#                           HELPER FUNCTIONS
###############################################################################

# Load CSV data into a DataFrame
@st.cache_data
def load_data():
    df = pd.read_csv('urldata.csv')  # Adjust path if needed
    return df

# Load pretrained models from disk and store in session state
@st.cache_resource
def load_pretrained_models(file_path="trained_models.pkl"):
    if "model_results" not in st.session_state:
        with open(file_path, 'rb') as f:
            st.session_state["model_results"] = pickle.load(f)
    return st.session_state["model_results"]

# Generate a Plotly confusion matrix chart
def plot_conf_matrix(y_true, y_pred, labels=("Legit", "Phish"), title="Confusion Matrix"):
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

# Store model metrics and results in a dictionary
def store_model_result(name, model, y_pred_train, y_pred_test, y_train, y_test):
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

# Train a Decision Tree model with given max depth
def train_decision_tree_model(X_train, y_train, X_test, y_test, max_depth):
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X_train, y_train)
    return store_model_result("Decision Tree", tree, tree.predict(X_train), tree.predict(X_test), y_train, y_test)

# Train a Random Forest model with given max depth and number of estimators
def train_random_forest_model(X_train, y_train, X_test, y_test, max_depth, n_estimators):
    forest = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    forest.fit(X_train, y_train)
    return store_model_result("Random Forest", forest, forest.predict(X_train), forest.predict(X_test), y_train, y_test)

# Train an MLP model with given alpha and hidden layer sizes
def train_mlp_model(X_train, y_train, X_test, y_test, alpha, hidden_layer_sizes):
    mlp = MLPClassifier(alpha=alpha, hidden_layer_sizes=hidden_layer_sizes, random_state=42)
    mlp.fit(X_train, y_train)
    return store_model_result("Multilayer Perceptrons", mlp, mlp.predict(X_train), mlp.predict(X_test), y_train, y_test)

# Train an XGBoost model with given learning rate, max depth, and number of estimators
def train_xgboost_model(X_train, y_train, X_test, y_test, learning_rate, max_depth, n_estimators):
    xgb = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    xgb.fit(X_train, y_train)
    return store_model_result("XGBoost", xgb, xgb.predict(X_train), xgb.predict(X_test), y_train, y_test)

# Train all models using provided training parameters and return results
def train_all_models(X_train, y_train, X_test, y_test,
                     dt_max_depth, rf_max_depth, rf_n_estimators,
                     mlp_alpha, mlp_hidden_layer_sizes,
                     xgb_learning_rate, xgb_max_depth, xgb_n_estimators):
    results = {}
    results["Decision Tree"] = train_decision_tree_model(X_train, y_train, X_test, y_test, dt_max_depth)
    results["Random Forest"] = train_random_forest_model(X_train, y_train, X_test, y_test, rf_max_depth, rf_n_estimators)
    results["Multilayer Perceptrons"] = train_mlp_model(X_train, y_train, X_test, y_test, mlp_alpha, mlp_hidden_layer_sizes)
    results["XGBoost"] = train_xgboost_model(X_train, y_train, X_test, y_test, xgb_learning_rate, xgb_max_depth, xgb_n_estimators)
    return results

# Generate a Graphviz tree graph for a Decision Tree model
def get_tree_graph(tree_model, feature_names):
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
#              PHISHING URL DETECTOR & FEATURE EXTRACTION FUNCTIONS
###############################################################################

# Calculate the Shannon entropy of a string
def calculate_entropy(s):
    if not s:
        return 0
    probs = [float(s.count(c)) / len(s) for c in dict.fromkeys(s)]
    return -sum(p * math.log(p, 2) for p in probs)

# Count subdomains in a given domain
def count_subdomains(domain):
    tokens = domain.split('.')
    return max(0, len(tokens) - 2)

# Count number of digits in a URL
def count_digits(url):
    return sum(c.isdigit() for c in url)

# Count special characters in a URL
def count_special_chars(url):
    return sum(1 for c in url if not c.isalnum() and c not in ".:/?-_")

# Compute ratio of uppercase letters in a URL
def uppercase_ratio(url):
    if len(url) == 0:
        return 0
    return sum(1 for c in url if c.isupper()) / len(url)

# Get the age of a domain using WHOIS lookup
def get_domain_age(domain):
    try:
        import whois
        w = whois.whois(domain)
        creation_date = w.creation_date
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

# Get HTTP response from a URL with a timeout
def get_http_response(url):
    try:
        resp = requests.get(url, timeout=5)
        return resp
    except RequestException:
        return ""

# Check for iframes in HTML response
def iframe_check(response):
    if not response or response == "":
        return 1
    return 0 if re.search(r"<iframe|<frameBorder", response.text, re.IGNORECASE) else 1

# Check for mouse-over events in HTML response
def mouse_over_check(response):
    if not response or response == "":
        return 1
    return 1 if re.search(r"<script.*onmouseover.*</script>", response.text, re.IGNORECASE | re.DOTALL) else 0

# Check for right-click disabling in HTML response
def right_click_check(response):
    if not response or response == "":
        return 1
    return 0 if re.search(r"event\.button\s*==\s*2", response.text) else 1

# Check if the page has too many redirects
def forwarding_check(response):
    if not response or response == "":
        return 1
    return 1 if len(response.history) > 2 else 0

# Determine if a URL is an IP address
def is_ip_address(url):
    try:
        ipaddress.ip_address(url)
        return True
    except:
        return False

# Convert extracted features to numeric values (skip non-numeric domain)
def convert_features_to_numeric(feat_list):
    if isinstance(feat_list[0], str):
        feat_list = feat_list[1:]
    return [float(val) if not isinstance(val, str) else 0.0 for val in feat_list]

# Extract a list of numeric features from a URL for prediction
def extract_features(url: str) -> list:
    domain = urlparse(url).netloc
    if re.match(r"^www\.", domain):
        domain = domain.replace("www.", "")
    ip_feat = 1 if is_ip_address(url) else 0
    at_feat = 1 if "@" in url else 0
    length_feat = 1 if len(url) >= 54 else 0
    depth_feat = sum(1 for part in urlparse(url).path.split('/') if part)
    redirect_feat = 1 if url.rfind('//') > 7 else 0
    https_in_domain_feat = 1 if 'https' in domain else 0
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
    url_entropy = calculate_entropy(url)
    domain_entropy = calculate_entropy(domain)
    subdomain_ct = count_subdomains(domain)
    digit_ct = count_digits(url)
    special_char_ct = count_special_chars(url)
    uppercase_rat = uppercase_ratio(url)
    domain_age = get_domain_age(domain)
    response = get_http_response(url)
    iframe_feat = iframe_check(response)
    mouse_feat = mouse_over_check(response)
    right_click_feat = right_click_check(response)
    forward_feat = forwarding_check(response)
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

# Set up page config and title
st.set_page_config(page_title="Phishing Detection App", layout="wide")
st.title("Phishing URL Detection — Full Analysis (Improved)")
st.markdown(
    """
    **New Update:**  
    We've now integrated improved WHOIS lookups, training parameter tuning, and individual model training options.
    """
)

# Load the dataset
df = load_data()

# Create tabs for different sections of the app
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
# 1. Data Overview Tab
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
        - `Having_IP`, `Have_At`, `URL_Length`, `URL_Depth`, `Redirection`, `HTTPS_in_Domain`, etc.

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
        2) Train classification models (Decision Tree, Random Forest, MLP, XGBoost) with custom parameters.  
        3) Compare model performance & interpret feature importances.  
        4) Provide a live URL checker using the trained/loaded models.
        """
    )

###############################################################################
# 2. EDA Tab
###############################################################################
with tab_eda:
    st.subheader("2. Exploratory Data Analysis (EDA)")
    # Prepare data for visualization by dropping non-numeric column if present
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
# 3. Train & Evaluate Models Tab
###############################################################################
with tab_modeling:
    st.subheader("3. Train & Evaluate Models")
    # Check if models already exist in session state
    if "model_results" in st.session_state:
        st.success("✅ Pretrained or previously trained models are available!")
    else:
        st.warning("No models found. Load or train models below.")

    # Prepare data for model training by dropping non-numeric column if necessary
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

    # Option to load models from disk
    col_load, _ = st.columns(2)
    with col_load:
        if st.button("Load Pretrained Models from Disk"):
            with st.spinner("Loading pretrained models..."):
                loaded_models_dict = load_pretrained_models("trained_models.pkl")
                model_results = {}
                for m_name, m_obj in loaded_models_dict.items():
                    # If model is stored in a dict, extract the actual model
                    actual_model = m_obj["model_object"] if isinstance(m_obj, dict) and "model_object" in m_obj else m_obj
                    y_pred_train = actual_model.predict(X_train)
                    y_pred_test = actual_model.predict(X_test)
                    model_results[m_name] = store_model_result(
                        m_name, actual_model, y_pred_train, y_pred_test, y_train, y_test
                    )
                st.session_state["model_results"] = model_results
                st.success("Loaded pretrained models from disk.")

    # Let user set training parameters for each model
    with st.expander("Training Parameters", expanded=True):
        st.markdown("### Set Training Parameters for Each Model")
        st.subheader("Decision Tree")
        dt_max_depth = st.number_input("Max Depth", min_value=1, value=20, key="dt_max_depth")
        st.subheader("Random Forest")
        rf_max_depth = st.number_input("Max Depth", min_value=1, value=40, key="rf_max_depth")
        rf_n_estimators = st.number_input("Number of Estimators", min_value=10, value=100, key="rf_n_estimators")
        st.subheader("MLP")
        mlp_alpha = st.number_input("Alpha", min_value=0.0001, value=0.001, format="%.4f", key="mlp_alpha")
        mlp_hidden_layer_sizes_str = st.text_input("Hidden Layer Sizes (comma separated)", value="200,200,200", key="mlp_hidden_layer_sizes")
        try:
            mlp_hidden_layer_sizes = tuple(int(x.strip()) for x in mlp_hidden_layer_sizes_str.split(",") if x.strip().isdigit())
        except:
            mlp_hidden_layer_sizes = (200, 200, 200)
        st.subheader("XGBoost")
        xgb_learning_rate = st.number_input("Learning Rate", min_value=0.001, value=0.4, format="%.3f", key="xgb_learning_rate")
        xgb_max_depth = st.number_input("Max Depth", min_value=1, value=42, key="xgb_max_depth")
        xgb_n_estimators = st.number_input("Number of Estimators", min_value=10, value=100, key="xgb_n_estimators")

    # Buttons to train each model individually
    st.markdown("### Train Models Individually")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Train Decision Tree"):
        with st.spinner("Training Decision Tree..."):
            dt_result = train_decision_tree_model(X_train, y_train, X_test, y_test, dt_max_depth)
            st.session_state.setdefault("model_results", {})["Decision Tree"] = dt_result
            st.success("Decision Tree trained!")
    if col2.button("Train Random Forest"):
        with st.spinner("Training Random Forest..."):
            rf_result = train_random_forest_model(X_train, y_train, X_test, y_test, rf_max_depth, rf_n_estimators)
            st.session_state.setdefault("model_results", {})["Random Forest"] = rf_result
            st.success("Random Forest trained!")
    if col3.button("Train MLP"):
        with st.spinner("Training MLP..."):
            mlp_result = train_mlp_model(X_train, y_train, X_test, y_test, mlp_alpha, mlp_hidden_layer_sizes)
            st.session_state.setdefault("model_results", {})["Multilayer Perceptrons"] = mlp_result
            st.success("MLP trained!")
    if col4.button("Train XGBoost"):
        with st.spinner("Training XGBoost..."):
            xgb_result = train_xgboost_model(X_train, y_train, X_test, y_test, xgb_learning_rate, xgb_max_depth, xgb_n_estimators)
            st.session_state.setdefault("model_results", {})["XGBoost"] = xgb_result
            st.success("XGBoost trained!")

    # Button to train all models together
    if st.button("Train All Models"):
        with st.spinner("Training all models..."):
            all_results = train_all_models(
                X_train, y_train, X_test, y_test,
                dt_max_depth, rf_max_depth, rf_n_estimators,
                mlp_alpha, mlp_hidden_layer_sizes,
                xgb_learning_rate, xgb_max_depth, xgb_n_estimators
            )
            st.session_state["model_results"] = all_results
            st.success("All models trained!")

    # Display the results of model training if available
    if "model_results" in st.session_state:
        st.write("### Model Results")
        for name, res in st.session_state["model_results"].items():
            with st.expander(f"{name} Results", expanded=False):
                st.write(f"**Accuracy (Train):** {res['acc_train']:.3f}")
                st.write(f"**Accuracy (Test):** {res['acc_test']:.3f}")
                st.write(f"**Precision (Train):** {res['prec_train']:.3f}")
                st.write(f"**Precision (Test):** {res['prec_test']:.3f}")
                st.plotly_chart(res["conf_matrix_fig"], use_container_width=True)
    else:
        st.info("No models loaded or trained yet. Use one of the above options.")

###############################################################################
# 4. Compare All Models Tab
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
# 5. Phishing URL Detector Tab (With Model Selection)
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
                st.write("Extracting advanced features...")
                model_obj = st.session_state["model_results"][selected_model]["model_object"]
                if "Domain" in df.columns:
                    feature_cols = list(df.drop(["Domain", "Label"], axis=1).columns)
                else:
                    feature_cols = list(df.drop("Label", axis=1).columns)
                features_vector = extract_features(url_input)
                features_df = pd.DataFrame([features_vector], columns=feature_cols)
                prediction = model_obj.predict(features_df)
                if prediction[0] == 1:
                    st.error(f"🚨 **Phishing Alert!** `{selected_model}` classifies this URL as phishing.")
                else:
                    st.success(f"✅ **Safe URL!** `{selected_model}` says this seems legit.")
        st.markdown(
            """
            **How It Works**:
            - An expanded feature set (domain age, lexical stats, etc.) is extracted.
            - The selected model then classifies the URL as either *Phishing* or *Legit*.
            """
        )