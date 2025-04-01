# pages/3_🛠️_Model_Training_&_Evaluation.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import (load_data, load_pretrained_models,
                   train_decision_tree_model, train_random_forest_model,
                   train_mlp_model, train_xgboost_model, store_model_result, safe_rerun)
import time

st.set_page_config(page_title="Train & Evaluate Models", page_icon="🛠️", layout="wide")

st.title("🛠️ Model Training & Evaluation")
st.markdown("Train various Machine Learning models or load pre-trained ones to see how well they detect phishing URLs.")
st.markdown("---")

# --- Load Data ---
df = load_data()

if df is not None:
    # --- Data Preparation ---
    st.markdown("### Data Preparation")
    # Drop non-numeric/ID columns, keep Label
    data_model = df.copy()
    potential_drop = ['Domain', 'URL', 'id'] # Add others if needed
    cols_to_drop = [col for col in potential_drop if col in data_model.columns]
    if cols_to_drop:
        data_model = data_model.drop(columns=cols_to_drop)

    # Handle potential NaN values (simple strategy: fill with 0)
    # Consider more sophisticated imputation if needed based on EDA
    if data_model.isnull().sum().sum() > 0:
         st.warning(f"Found {data_model.isnull().sum().sum()} missing values. Filling with 0 for training.", icon="⚠️")
         data_model = data_model.fillna(0)

    X = data_model.drop("Label", axis=1)
    y = data_model["Label"]
    feature_names = list(X.columns) # Store feature names for later use

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y # Stratify for balanced splits
    )
    st.write(f"Data split into **Training Set ({X_train.shape[0]} samples)** and **Test Set ({X_test.shape[0]} samples)**.")
    st.markdown("---")

    # --- Initialize Session State for Results ---
    if "model_results" not in st.session_state:
        st.session_state["model_results"] = {}
        st.session_state["trained_model_params"] = {} # Store params used for trained models

    # --- Load Pre-trained Models Option ---
    st.markdown("### Option 1: Load Pre-trained Models")
    st.info("Load models that were trained beforehand (e.g., with optimized parameters).")
    if st.button("Load Pre-trained Models from File", key="load_pretrained"):
        pretrained_data = load_pretrained_models("trained_models.pkl") # Load from utils
        if pretrained_data:
            with st.spinner("Processing pre-trained models..."):
                # Assuming pretrained_data is a dict like {'Model Name': model_object}
                # We need to re-calculate metrics on the current train/test split
                temp_results = {}
                for name, model in pretrained_data.items():
                    if hasattr(model, 'predict'):
                        y_pred_train_load = model.predict(X_train)
                        y_pred_test_load = model.predict(X_test)
                        temp_results[name] = store_model_result(
                            name, model, y_pred_train_load, y_pred_test_load, y_train, y_test
                        )
                        # Try to guess params (or load them if saved) - placeholder
                        st.session_state["trained_model_params"][name] = "Pre-trained (Params not recorded here)"
                    else:
                        st.warning(f"Skipping '{name}' from pretrained file as it doesn't seem to be a valid model object.")

                st.session_state["model_results"].update(temp_results) # Add loaded results
            st.success(f"Loaded and evaluated {len(temp_results)} pre-trained models on the current data split!")
            time.sleep(1) # Give time to read message
            safe_rerun() # Rerun to update display below

    st.markdown("---")
    st.markdown("### Option 2: Train a New Model")

    # --- Model Selection and Hyperparameters ---
    model_option = st.selectbox(
        "Choose a Model to Train:",
        ("Decision Tree", "Random Forest", "MLP", "XGBoost"),
        key="model_select"
    )

    train_button_pressed = False
    st.write(f"**Configure Hyperparameters for {model_option}:**")
    params = {}

    if model_option == "Decision Tree":
        params['max_depth'] = st.slider("Max Depth", 1, 50, 10, key="dt_depth",
                                       help="Maximum depth of the tree. Deeper trees can overfit.")
        if st.button(f"Train {model_option}", key="train_dt"):
             train_button_pressed = True
             model_func = train_decision_tree_model
             train_args = (X_train, y_train, X_test, y_test, params['max_depth'])

    elif model_option == "Random Forest":
        params['max_depth'] = st.slider("Max Depth", 1, 50, 20, key="rf_depth",
                                       help="Maximum depth of individual trees in the forest.")
        params['n_estimators'] = st.slider("Number of Estimators (Trees)", 10, 300, 100, step=10, key="rf_est",
                                          help="Number of trees in the forest. More trees generally improve performance but increase training time.")
        if st.button(f"Train {model_option}", key="train_rf"):
             train_button_pressed = True
             model_func = train_random_forest_model
             train_args = (X_train, y_train, X_test, y_test, params['max_depth'], params['n_estimators'])

    elif model_option == "MLP":
        params['alpha'] = st.slider("Alpha (L2 Regularization)", 0.0001, 0.1, 0.001, format="%.4f", key="mlp_alpha",
                                  help="Regularization strength. Helps prevent overfitting.")
        hidden_layer_str = st.text_input("Hidden Layer Sizes (comma-separated)", "100,50", key="mlp_layers",
                                         help="Number of neurons in each hidden layer. E.g., '100' for one layer, '100,50' for two.")
        try:
            params['hidden_layer_sizes'] = tuple(int(x.strip()) for x in hidden_layer_str.split(',') if x.strip())
        except ValueError:
            st.error("Invalid format for Hidden Layer Sizes. Please use comma-separated integers (e.g., 100,50). Using default (100,50).")
            params['hidden_layer_sizes'] = (100, 50)
        if st.button(f"Train {model_option}", key="train_mlp"):
             train_button_pressed = True
             model_func = train_mlp_model
             train_args = (X_train, y_train, X_test, y_test, params['alpha'], params['hidden_layer_sizes'])

    elif model_option == "XGBoost":
        params['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01, key="xgb_lr",
                                            help="Step size shrinkage to prevent overfitting. Lower values usually require more estimators.")
        params['max_depth'] = st.slider("Max Depth", 1, 20, 6, key="xgb_depth",
                                       help="Maximum depth of individual trees.")
        params['n_estimators'] = st.slider("Number of Estimators (Rounds)", 50, 500, 100, step=10, key="xgb_est",
                                          help="Number of boosting rounds (trees).")
        if st.button(f"Train {model_option}", key="train_xgb"):
             train_button_pressed = True
             model_func = train_xgboost_model
             train_args = (X_train, y_train, X_test, y_test, params['learning_rate'], params['max_depth'], params['n_estimators'])

    # --- Training Execution ---
    if train_button_pressed:
        st.markdown("---")
        st.info(f"Training {model_option}... Please wait.")
        progress_bar = st.progress(0, text="Initializing...")
        try:
            # Pass progress bar to training function
            result = model_func(*train_args, progress_bar=progress_bar)
            st.session_state["model_results"][model_option] = result
            st.session_state["trained_model_params"][model_option] = params # Store the params used
            st.success(f"{model_option} trained successfully!")
            time.sleep(1) # Let user see success message
            safe_rerun() # Rerun to update display
        except Exception as e:
            progress_bar.empty() # Remove progress bar on error
            st.error(f"An error occurred during {model_option} training: {e}")
            # Optionally clear the potentially failed result
            if model_option in st.session_state["model_results"]:
                 del st.session_state["model_results"][model_option]
            if model_option in st.session_state["trained_model_params"]:
                 del st.session_state["trained_model_params"][model_option]


    st.markdown("---")
    # --- Display Results ---
    st.subheader("📊 Current Model Results")
    if not st.session_state["model_results"]:
        st.warning("No models trained or loaded yet in this session.")
    else:
        # Get feature names if available
        trained_feature_names = list(X_train.columns) if 'X_train' in locals() else None

        for name, res in st.session_state["model_results"].items():
            with st.expander(f"Show Results for: **{name}**", expanded=False):
                st.write(f"**Parameters Used:** `{st.session_state['trained_model_params'].get(name, 'N/A')}`")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Test Accuracy", value=f"{res['acc_test']:.3f}")
                    st.metric(label="Test Precision", value=f"{res['prec_test']:.3f}")
                    st.metric(label="Train Accuracy", value=f"{res['acc_train']:.3f}")
                    st.metric(label="Train Precision", value=f"{res['prec_train']:.3f}")
                with col2:
                    st.plotly_chart(res["conf_matrix_fig"], use_container_width=True)

                # Display feature importance if available and requested
                # Moved to Comparison page for better flow

else:
    st.error("Failed to load data. Cannot train or evaluate models.")

st.markdown("---")