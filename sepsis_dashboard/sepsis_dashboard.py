import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

# --------------------------
# 1. Load Model and Data
# --------------------------
st.title("Sepsis Patient Risk Dashboard")
st.markdown(
    """
    #### Application Overview
    This application demonstrates a **Sepsis Patient Risk** model using a **Stacked Ensemble** approach. You can:
    
    - Filter patients by different clinical features (using the sidebar).
    - Examine the **predicted probability of 30-day mortality**.
    - Explore **SHAP** explanations for individual predictions.
    - Review **static performance metrics** on the hold-out test set.
    
    #### Instructions
    1. **Filter Patients**: Use the sidebar to select sepsis types and adjust clinical feature ranges.
    2. **Explore Predictions**: Observe the **Summary Metrics** section for filtered results.
    3. **SHAP Visualizations**: Use the dropdown to analyze individual predictions or explore the global feature importance.
    4. **Performance Metrics**: Scroll down to see test performance and risk distributions.
    """
)

# Load pre-trained model and data
model = joblib.load("sepsis_dashboard/stacking_model.pkl")
file_path = r"sepsis_dashboard/sepsis.csv"
df = pd.read_csv(file_path, encoding='utf-8', escapechar='\\')

y = pd.read_csv("sepsis_dashboard/thirtyday_expire_flag_labels.csv")
shap_values_rf_all = pd.read_csv("sepsis_dashboard/shap_vals_rf_sepsis.csv", index_col=0)
shap_values_rf_test = pd.read_csv("sepsis_dashboard/shap_vals_rf_sepsis_X_test.csv", index_col=0)
X_test = pd.read_csv("sepsis_dashboard/X_test_sepsis.csv")

sepsis_types = ['sepsis_angus', 'sepsis_martin', 'sepsis_explicit', 
                'septic_shock_explicit', 'severe_sepsis_explicit', 
                'sepsis_nqf', 'sepsis_cdc', 'sepsis_cdc_simple']

top_features = ['urineoutput', 'lods', 'age', 'elixhauser_hospital',
                'resprate_mean', 'lactate_mean', 'sofa',
                'bun_mean', 'aniongap_max', 'sysbp_mean']

explainer_rf_base_value = 0.49729466033467146

# --------------------------
# 2. Sidebar Filters
# --------------------------
df = df[df[sepsis_types].sum(axis=1) > 0]
df = df.dropna(subset=top_features)

st.sidebar.header("Filters")
selected_sepsis = st.sidebar.multiselect("Select Sepsis Types:", sepsis_types, default=["sepsis_angus"])
selected_features = st.sidebar.multiselect("Select Clinical Features:", top_features, default=["urineoutput", "age"])

feature_ranges = {}
for feature in selected_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    feature_ranges[feature] = st.sidebar.slider(
        f"{feature} range:", min_value=min_val, max_value=max_val, value=(min_val, max_val)
    )

# --------------------------
# 3. Filter Data
# --------------------------
sepsis_filter = df[selected_sepsis].sum(axis=1) > 0
filtered_df = df[sepsis_filter]
for feature in selected_features:
    min_val, max_val = feature_ranges[feature]
    filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]

filtered_df = filtered_df.reset_index(drop=True)
common_indices = shap_values_rf_all.index.intersection(filtered_df.index)
shap_values_rf_all = shap_values_rf_all.loc[common_indices]
filtered_df = filtered_df.loc[common_indices]

if not filtered_df.empty:
    shap_values_filtered = shap_values_rf_all.loc[common_indices]
    filtered_df['predicted_probability'] = shap_values_filtered.sum(axis=1) + explainer_rf_base_value
    filtered_df = filtered_df.sort_values(by='predicted_probability', ascending=False)
    shap_values_filtered = shap_values_filtered.loc[filtered_df.index]

    # Summary Metrics
    st.markdown("### Summary Metrics")
    total_patients = len(filtered_df)
    died_count = filtered_df["thirtyday_expire_flag"].sum()
    predicted_deaths = (filtered_df['predicted_probability'] >= 0.5).sum()
    st.markdown(f"""
    - **Total Patients Filtered**: {total_patients}
    - **Actual Deaths**: {died_count} ({(died_count / total_patients) * 100:.2f}%)
    - **Predicted Deaths**: {predicted_deaths} ({(predicted_deaths / total_patients) * 100:.2f}%)
    - **Average Predicted Probability of Death**: {filtered_df['predicted_probability'].mean():.2f}
    """)

# --------------------------
# 4. SHAP Force Plot
# --------------------------
st.markdown("### SHAP Force Plot for Individual Predictions")
if not filtered_df.empty:
    selected_patient_index = st.selectbox(
        "Select a Patient for Force Plot:",
        filtered_df.index,
        format_func=lambda x: f"Patient {x} (Predicted Probability: {filtered_df.loc[x, 'predicted_probability']:.2f})"
    )

    selected_shap_values = shap_values_filtered.loc[selected_patient_index].values
    selected_features = filtered_df.loc[selected_patient_index, top_features]
    force_plot = shap.plots.force(explainer_rf_base_value, selected_shap_values, features=selected_features)
    shap.save_html("force_plot.html", force_plot)
    with open("force_plot.html", "r", encoding="utf-8") as f:
        force_plot_html = f.read()
    with st.container():
        st.components.v1.html(force_plot_html, height=500, scrolling=True)


# --------------------------
# 5. Test Performance Metrics (Static)
# --------------------------
with open("sepsis_dashboard/test_metrics.json", "r") as f:
    test_metrics = json.load(f)
threshold = test_metrics["threshold"]
f1_value = test_metrics["f1_score"]
classification_rpt = test_metrics["classification_report"]
conf_matrix = test_metrics["confusion_matrix"]
roc_auc_value = test_metrics["roc_auc"]
st.markdown("### Model Performance (Test Set)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Threshold", value=f"{threshold}")
with col2:
    st.metric(label="F1 Score", value=f"{f1_value:.3f}")
with col3:
    st.metric(label="ROC AUC", value=f"{roc_auc_value:.3f}")

# Convert dict to DataFrame
report_df = pd.DataFrame(classification_rpt).transpose()

st.markdown("### Classification Report")
st.table(report_df)

st.markdown("### Confusion Matrix (Heatmap)")
fig, ax = plt.subplots(figsize=(3, 3))  # Adjusted figure size to make it more compact
sns.heatmap(
    test_metrics["confusion_matrix"], 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    cbar=False, 
    xticklabels=["Pred 0", "Pred 1"], 
    yticklabels=["Actual 0", "Actual 1"], 
    annot_kws={"size": 8},  # Reduced font size for annotations
    square=True,  # Ensure square cells for better visibility
    linewidths=0.5,  # Add thin lines between boxes for clarity
    ax=ax
)
ax.set_title("Confusion Matrix", fontsize=10)  # Reduced title font size
ax.tick_params(axis='both', labelsize=8)  # Adjusted axis label font size
st.pyplot(fig)

# --------------------------
# 7. Patient Risk Distribution
# --------------------------
st.subheader("Patient Risk Distribution")
fig = px.histogram(
    filtered_df, x="predicted_probability", nbins=20,
    title="Distribution of Predicted Probabilities",
    labels={"predicted_probability": "Predicted Probability of Death"},
    color_discrete_sequence=["#636EFA"]
)
st.plotly_chart(fig)

# --------------------------
# 8. Show Filtered Data
# --------------------------
if st.checkbox("Show Filtered Data"):
    st.subheader("Filtered Data")
    st.dataframe(filtered_df)
