# pages/2_📈_EDA.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import load_data # Import from utils

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis (EDA)")
st.markdown("Let's visualize the data to understand patterns and relationships.")
st.markdown("---")

# --- Load Data ---
df = load_data()

if df is not None:
    # Drop non-numeric columns unlikely to be useful for general EDA (like raw domain string if present)
    # Keep 'Label' for analysis
    df_eda = df.copy()
    potential_drop = ['Domain', 'URL', 'id'] # Add other non-numeric/ID columns if they exist
    cols_to_drop = [col for col in potential_drop if col in df_eda.columns]
    if cols_to_drop:
        df_eda = df_eda.drop(columns=cols_to_drop)

    st.markdown("### 🎯 Label Distribution (Phishing vs. Legit)")
    label_counts = df_eda['Label'].value_counts()
    label_names = {0: "Legit (0)", 1: "Phish (1)"}
    label_counts.index = label_counts.index.map(label_names)

    # Use a Pie Chart for binary label distribution
    fig_pie = px.pie(values=label_counts.values, names=label_counts.index,
                     title="Distribution of Phishing vs. Legitimate URLs",
                     hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pie.update_traces(textinfo='percent+label', pull=[0, 0.05]) # Pull the phishing slice slightly
    fig_pie.update_layout(legend_title_text='URL Type', title_x=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.info(f"The dataset contains {label_counts.get('Legit (0)', 0)} legitimate URLs and {label_counts.get('Phish (1)', 0)} phishing URLs.")

    st.markdown("---")
    st.markdown("### Numeric Feature Distributions")

    numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label') # Exclude label from feature distribution analysis

    if numeric_cols:
        col1, col2 = st.columns([1, 3]) # Layout: Selector | Plot
        with col1:
            chosen_num_col = st.selectbox("Select a Numeric Feature:",
                                          options=numeric_cols,
                                          index=0 if 'URL_Length' not in numeric_cols else numeric_cols.index('URL_Length'), # Default selection
                                          help="Choose a feature to see its distribution.")
        with col2:
            # Use Histogram for distribution
            fig_hist = px.histogram(df_eda, x=chosen_num_col, color='Label', # Color by label
                                    marginal="box", # Add box plots to see spread
                                    barmode='overlay', # Overlay histograms
                                    title=f"Distribution of '{chosen_num_col}' by Label",
                                    labels={'Label': 'URL Type (0: Legit, 1: Phish)'},
                                    opacity=0.7)
            fig_hist.update_layout(title_x=0.5)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Key Takeaway based on selected feature (Example)
            if chosen_num_col == 'Domain_Age' and 'Domain_Age' in df_eda.columns:
                median_age_phish = df_eda[df_eda['Label'] == 1]['Domain_Age'].median()
                median_age_legit = df_eda[df_eda['Label'] == 0]['Domain_Age'].median()
                st.info(f"**Takeaway:** Phishing sites often have a much lower median domain age (around {median_age_phish:.0f} days, if available) compared to legitimate sites (around {median_age_legit:.0f} days). Values of -1 indicate failed lookups.")
            elif chosen_num_col == 'URL_Length':
                 st.info("**Takeaway:** Observe if one category (Phishing or Legit) tends to have longer URLs (higher values).")


    else:
        st.warning("No numeric columns found (excluding 'Label') for distribution analysis.")

    st.markdown("---")
    st.markdown("### Binary Feature Analysis")

    # Identify binary columns (0/1 values, excluding Label)
    binary_cols = [col for col in df_eda.columns if df_eda[col].nunique() == 2 and col != 'Label' and set(df_eda[col].unique()) <= {0, 1, -1}] # Allow -1 too

    if binary_cols:
        chosen_bin_col = st.selectbox("Select a Binary Feature:",
                                      options=binary_cols,
                                      index=0 if 'Having_IP' not in binary_cols else binary_cols.index('Having_IP'), # Default
                                      help="Analyze features with two main values (like Yes/No flags).")

        # Create a stacked bar chart
        counts = df_eda.groupby([chosen_bin_col, 'Label']).size().unstack(fill_value=0)
        counts.index = counts.index.map({0: f"{chosen_bin_col}=No", 1: f"{chosen_bin_col}=Yes", -1: f"{chosen_bin_col}=N/A"}) # Map index values to readable names
        counts.columns = counts.columns.map({0: "Legit", 1: "Phish"}) # Map columns to readable names

        fig_stacked = px.bar(counts, barmode='stack',
                             title=f"'{chosen_bin_col}' vs. URL Type",
                             labels={'value': 'Number of URLs', 'variable': 'URL Type', 'index': f'Value of {chosen_bin_col}'},
                             text_auto=True, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_stacked.update_layout(title_x=0.5)
        st.plotly_chart(fig_stacked, use_container_width=True)

        # Key Takeaway (Example)
        if chosen_bin_col == 'Having_IP':
             st.info("**Takeaway:** Check if the proportion of Phishing URLs is higher when `Having_IP` is Yes (1). This would suggest IP-based URLs are a strong indicator.")
        elif chosen_bin_col == 'Prefix/Suffix':
             st.info("**Takeaway:** See if domains containing hyphens (`Prefix/Suffix` = Yes/1) are more commonly associated with Phishing or Legit URLs.")

    else:
         st.warning("No binary (0/1) columns found (excluding 'Label') for this analysis.")


    st.markdown("---")
    st.markdown("### Feature Correlation Heatmap")
    # Select only numeric columns for correlation
    numeric_df_corr = df_eda.select_dtypes(include=np.number)

    if len(numeric_df_corr.columns) > 1:
        corr = numeric_df_corr.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", # Format to 2 decimal places
                             aspect="auto",
                             color_continuous_scale='RdBu_r', # Red-Blue diverging scale
                             title="Correlation Matrix of Numeric Features")
        fig_corr.update_layout(title_x=0.5)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info(
            """
            **How to Read:**
            - **Closer to +1 (Red):** Strong positive correlation (features tend to increase together).
            - **Closer to -1 (Blue):** Strong negative correlation (one increases as the other decreases).
            - **Closer to 0 (White):** Weak or no linear correlation.
            High correlation between features might indicate redundancy. Correlation with the `Label` column shows how strongly a feature is linearly related to phishing (though models capture non-linear relationships too).
            """
        )
    else:
        st.warning("Not enough numeric columns to compute a correlation heatmap.")

else:
    st.error("Failed to load data. Cannot perform EDA.")

st.markdown("---")