import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt

# --------------------------
# 1. Load Model and Data
# --------------------------
st.title("Sepsis Patient Risk Dashboard")
st.markdown(
    """
    This application demonstrates a **Sepsis Patient Risk** model using a
    **Stacked Ensemble** approach. You can:
    
    - Filter patients by different clinical features (using the sidebar).
    - Examine the **predicted probability of 30-day mortality**.
    - Explore **SHAP** explanations for individual predictions.
    - Review **static performance metrics** on the hold-out test set.
    
    **How to Use**:
    1. Select one or more Sepsis Types in the sidebar.
    2. Choose the clinical features you want to filter by, and adjust their numerical ranges.
    3. Observe the **Summary Metrics** section for how many patients are filtered, the actual and predicted deaths, and the average predicted death probability.
    4. For individual explanations, pick a patient from the dropdown in the **SHAP Force Plot** section.
    5. Explore the **SHAP Summary Plot (Global)** for overall feature importance on the test set (optional).
    6. Scroll down to see the **Patient Risk Distribution** and optionally display the **Filtered Data**.
    """
)
# Load pre-trained model and data
model = joblib.load("sepsis_dashboard/stacking_model.pkl")
file_path = r"sepsis_dashboard/sepsis.csv"
df = pd.read_csv(file_path, encoding='utf-8', escapechar='\\')

y = pd.read_csv("sepsis_dashboard/thirtyday_expire_flag_labels.csv")
# Load SHAP values
shap_values_rf_all = pd.read_csv("sepsis_dashboard/shap_vals_rf_sepsis.csv", index_col=0)
shap_values_rf_test = pd.read_csv("sepsis_dashboard/shap_vals_rf_sepsis_X_test.csv", index_col=0)
X_test= pd.read_csv("sepsis_dashboard/X_test_sepsis.csv")
# Sepsis types and top features
sepsis_types = [
    'sepsis_angus', 'sepsis_martin', 'sepsis_explicit', 
    'septic_shock_explicit', 'severe_sepsis_explicit', 
    'sepsis_nqf', 'sepsis_cdc', 'sepsis_cdc_simple'
]

top_features = [
    'urineoutput', 'lods', 'age', 'elixhauser_hospital',
    'resprate_mean', 'lactate_mean', 'sofa',
    'bun_mean', 'aniongap_max', 'sysbp_mean'
]

# Base value from SHAP explainer
explainer_rf_base_value = 0.49729466033467146
  # Adjust as per your model's SHAP base value.

# --------------------------
# 2. Sidebar Filters
# --------------------------
df = df[df[sepsis_types].sum(axis=1) > 0]
# Drop rows where top features have null values

df = df.dropna(subset=top_features)

#st.write("Initial shape:", df.shape)

#st.sidebar.header("Filters")

# Select Sepsis Types
selected_sepsis = st.sidebar.multiselect(
    "Select Sepsis Types:",
    sepsis_types,
    default=["sepsis_angus"]
)

# Select Clinical Features
selected_features = st.sidebar.multiselect(
    "Select Clinical Features:",
    top_features,
    default=["urineoutput", "age"]
)

# Feature ranges
feature_ranges = {}
for feature in selected_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    
    # Default range includes all values
    feature_ranges[feature] = st.sidebar.slider(
        f"{feature} range:",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)  # Use the entire range by default
    )



# --------------------------
# 3. Filter Data
# --------------------------

# Filter rows based on sepsis types
sepsis_filter = df[selected_sepsis].sum(axis=1) > 0

filtered_df = df[sepsis_filter]

#st.write("Shape after filtering for sepsis types:", filtered_df.shape)

# Apply filters for clinical features

for feature in selected_features:
    min_val, max_val = feature_ranges[feature]
    pre_filter_shape = filtered_df.shape
    filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]
    #st.write(f"Filtered {feature}: {pre_filter_shape[0] - filtered_df.shape[0]} rows excluded")


#st.write("Shape after applying clinical feature ranges:", filtered_df.shape)

filtered_df = filtered_df.reset_index(drop=True)

#st.write("Filtered_df indices (before alignment):", filtered_df.index)
#st.write("SHAP values indices:", shap_values_rf_all.index)
common_indices = shap_values_rf_all.index.intersection(filtered_df.index)

# Subset both SHAP values and filtered_df
shap_values_rf_all = shap_values_rf_all.loc[common_indices]
filtered_df = filtered_df.loc[common_indices]

# Debugging: Check shapes
#st.write("Shape of shap_values_rf_all after alignment:", shap_values_rf_all.shape)
#st.write("Shape of filtered_df after alignment:", filtered_df.shape)
#common_indices = shap_values_rf_all.index.intersection(filtered_df.index)

# Subset both filtered_df and SHAP values
#filtered_df = filtered_df.loc[common_indices]


if not filtered_df.empty:
    shap_values_filtered = shap_values_rf_all.loc[common_indices]

    # Compute predicted probabilities using SHAP values
    filtered_df['predicted_probability'] = shap_values_filtered.sum(axis=1) + explainer_rf_base_value

    # Sort by predicted probability in descending order
    filtered_df = filtered_df.sort_values(by='predicted_probability', ascending=False)

    # Ensure SHAP values are sorted to match the filtered_df
    shap_values_filtered = shap_values_filtered.loc[filtered_df.index]


    # Display survival stats
    total_patients = len(filtered_df)
    died_count = filtered_df["thirtyday_expire_flag"].sum()
    predicted_deaths = (filtered_df['predicted_probability'] >= 0.5).sum()

    st.subheader("Summary Metrics")
    st.write(f"**Total Patients Filtered**: {total_patients}")
    st.write(f"**Actual Deaths**: {died_count} ({(died_count / total_patients) * 100:.2f}%)")
    st.write(f"**Predicted Deaths**: {predicted_deaths} ({(predicted_deaths / total_patients) * 100:.2f}%)")
    st.write(f"**Average Predicted Probability of Death**: {filtered_df['predicted_probability'].mean():.2f}")
    #st.write(common_indices.shape)


# --------------------------
# 5. SHAP Force Plot
# --------------------------

st.subheader("SHAP Force Plot for Individual Predictions")

if not filtered_df.empty:
    # Patient selection dropdown
    selected_patient_index = st.selectbox(
        "Select a Patient for Force Plot:",
        filtered_df.index,
        format_func=lambda x: f"Patient {x} (Predicted Probability: {filtered_df.loc[x, 'predicted_probability']:.2f})"
    )

    # Display SHAP force plot for the selected patient
    selected_shap_values = shap_values_filtered.loc[selected_patient_index].values
    selected_features = filtered_df.loc[selected_patient_index, top_features]

    # Create the force plot using `shap.plots.force`
    force_plot = shap.plots.force(
        base_value=explainer_rf_base_value,
        shap_values=selected_shap_values,
        features=selected_features
    )

    # Save the force plot as HTML
    shap.save_html("force_plot.html", force_plot)

    # Display the HTML in Streamlit
    with open("force_plot.html", "r", encoding="utf-8") as f:
        force_plot_html = f.read()

    st.components.v1.html(force_plot_html, height=500)

# --------------------------
# 5. Test Performance Metrics (static)
# --------------------------
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json
import seaborn as sns 

# Load the JSON file
with open("sepsis_dashboard/test_metrics.json", "r") as f:
    test_metrics = json.load(f)


threshold = test_metrics["threshold"]
f1_value = test_metrics["f1_score"]
classification_rpt = test_metrics["classification_report"]
conf_matrix = test_metrics["confusion_matrix"]
roc_auc_value = test_metrics["roc_auc"]
st.title("Model Performance (Test Set)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Threshold", value=f"{threshold}")
with col2:
    st.metric(label="F1 Score", value=f"{f1_value:.3f}")
with col3:
    st.metric(label="ROC AUC", value=f"{roc_auc_value:.3f}")

# Convert dict to DataFrame
report_df = pd.DataFrame(classification_rpt).transpose()

st.subheader("Classification Report")
st.table(report_df)

st.subheader("Confusion Matrix (Heatmap)")

fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

# --------------------------
# 6. SHAP Summary Plot
# --------------------------

if st.checkbox("Show SHAP Summary Plot (Global)"):
    st.subheader("SHAP Summary Plot (Global)")
    st.write("The SHAP summary plot shows the global importance of features for the test dataset.")

    # Generate SHAP summary plot
    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values_rf_test.values,
        X_test,
        feature_names=top_features,
        show=False
    )
    st.pyplot(fig)

# --------------------------
# 7. Patient Risk Distribution
# --------------------------

st.subheader("Patient Risk Distribution")
fig = px.histogram(
    filtered_df,
    x="predicted_probability",
    nbins=20,
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
