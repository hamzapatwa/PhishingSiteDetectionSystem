# pages/4_⚖️_Model_Comparison.py
import streamlit as st
import pandas as pd
from utils import plot_feature_importance, plot_model_comparison, load_data # Import helpers
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Model Comparison", page_icon="⚖️", layout="wide")

st.title("⚖️ Model Comparison")
st.markdown("Compare the performance of all trained/loaded models side-by-side.")
st.markdown("---")

# Check if results exist in session state
if "model_results" not in st.session_state or not st.session_state["model_results"]:
    st.warning("No models have been trained or loaded in the 'Train & Evaluate' tab yet. Please go there first.", icon="⚠️")
else:
    model_results = st.session_state["model_results"]

    # --- Performance Summary Table ---
    st.subheader("📈 Performance Metrics Summary")
    summary_data = []
    for name, v in model_results.items():
        summary_data.append({
            "ML Model": name,
            "Test Accuracy": v['acc_test'],
            "Test Precision": v['prec_test'],
            # --- ADD RECALL TO SUMMARY DATA ---
            "Test Recall": v['recall_test'],
            # --- END ADD RECALL ---
            "Train Accuracy": v['acc_train'],
            "Train Precision": v['prec_train'],
            # --- ADD RECALL TO SUMMARY DATA ---
            "Train Recall": v['recall_train']
            # --- END ADD RECALL ---
        })

    summary_df = pd.DataFrame(summary_data)

    # Define columns order for display
    display_cols = ["ML Model", "Test Accuracy", "Test Precision", "Test Recall",
                    "Train Accuracy", "Train Precision", "Train Recall"]
    summary_df = summary_df[display_cols] # Reorder columns

    # Sort by Test Accuracy then Test Precision (descending) - keeping this sort order for now
    summary_df = summary_df.sort_values(by=["Test Accuracy", "Test Precision"], ascending=False).reset_index(drop=True)

    # Display styled dataframe
    # --- ADD RECALL FORMATTING ---
    st.dataframe(summary_df.style.format({
        "Train Accuracy": "{:.3f}", "Test Accuracy": "{:.3f}",
        "Train Precision": "{:.3f}", "Test Precision": "{:.3f}",
        "Train Recall": "{:.3f}", "Test Recall": "{:.3f}" # <-- Added recall format
    }).highlight_max(subset=['Test Accuracy', 'Test Precision', 'Test Recall'], color='lightgreen', axis=0), # Highlight max recall too
                 use_container_width=True)
    # --- END ADD RECALL FORMATTING ---

    # --- Identify and Highlight Best Model ---
    if not summary_df.empty:
        best_model_name = summary_df.iloc[0]["ML Model"]
        best_test_acc = summary_df.iloc[0]["Test Accuracy"]
        best_test_prec = summary_df.iloc[0]["Test Precision"]
        # --- GET BEST RECALL ---
        best_test_recall = summary_df.iloc[0]["Test Recall"]
        # --- UPDATE SUCCESS MESSAGE ---
        st.success(f"🏆 **Top Model (sorted by Test Accuracy & Precision):** `{best_model_name}` (Accuracy: {best_test_acc:.3f}, Precision: {best_test_prec:.3f}, Recall: {best_test_recall:.3f})")
        # --- UPDATE INTERPRETATION NOTES ---
        st.markdown(
             """
            **Interpretation:**
            - **High Test Accuracy, Precision, and Recall** are generally desired.
                - **Accuracy:** Overall correctness.
                - **Precision:** Of the URLs predicted as phishing, how many actually were? (Minimizes false positives)
                - **Recall:** Of all the actual phishing URLs, how many did the model find? (Minimizes false negatives - important for not missing threats!)
            - Compare **Train vs. Test** scores: A large gap might indicate **overfitting**.
            """
        )
        # --- END UPDATES ---
    else:
         st.warning("Summary table is empty.")


    st.markdown("---")
    # --- Visual Comparison Chart ---
    st.subheader("📊 Visual Performance Comparison")
    if not summary_df.empty:
        fig_compare = plot_model_comparison(summary_df)
        st.plotly_chart(fig_compare, use_container_width=True)
    else:
        st.info("No data to plot for comparison.")

    st.markdown("---")
    # --- Feature Importances ---
    st.subheader("🔑 Feature Importances")
    st.info("See which features the models rely on most for their predictions (available for Tree-based models).")

    # Need feature names - try to get them from a prepared dataset split if available
    # This requires data loading and splitting again, or storing X_train columns in session state
    df_fi = load_data()
    feature_names_fi = []
    if df_fi is not None:
        # Simplified data prep just to get columns - mirror Train page logic
        data_model_fi = df_fi.copy()
        potential_drop = ['Domain', 'URL', 'id', 'Label']
        cols_to_drop = [col for col in potential_drop if col in data_model_fi.columns]
        if cols_to_drop:
            data_model_fi = data_model_fi.drop(columns=cols_to_drop)
        if data_model_fi.isnull().sum().sum() > 0:
            data_model_fi = data_model_fi.fillna(0) # Consistent handling
        feature_names_fi = list(data_model_fi.columns)


    # Models that typically support feature importance
    importance_models = [name for name in model_results if name in ["Decision Tree", "Random Forest", "XGBoost"]]

    if not importance_models:
        st.warning("No models supporting feature importance (Decision Tree, Random Forest, XGBoost) were found in the results.")
    elif not feature_names_fi:
         st.warning("Could not retrieve feature names needed to display feature importances.")
    else:
        model_choice_fi = st.selectbox(
            "Select Model for Feature Importance:",
            options=importance_models,
            key="fi_select"
        )

        if model_choice_fi in model_results:
            selected_model_obj = model_results[model_choice_fi]["model_object"]
            fig_fi = plot_feature_importance(selected_model_obj, feature_names_fi, model_choice_fi)
            if fig_fi:
                st.plotly_chart(fig_fi, use_container_width=True)
                st.info(
                    f"""
                    **Interpretation:** Higher bars indicate features that the **{model_choice_fi}** model found more influential
                    in distinguishing between phishing and legitimate URLs. This helps understand the model's decision-making process.
                    """
                )
        else:
             st.error(f"Selected model '{model_choice_fi}' not found in results.")


st.markdown("---")