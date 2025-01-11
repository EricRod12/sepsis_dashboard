import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler

# --------------------------
# 1. Load Model and Data
# --------------------------

model = joblib.load("C:/Users/ericr/Downloads/stacking_model.pkl")  # Replace with your actual model path

file_path = r"C:/Users/ericr/Downloads/sepsis.csv"
df = pd.read_csv(file_path)

scaler = joblib.load("C:/Users/ericr/Downloads/scaler.pkl")         

# Sepsis types and top features
sepsis_types = ['sepsis_angus', 'sepsis_martin', 'sepsis_explicit', 'septic_shock_explicit', 'severe_sepsis_explicit', 'sepsis_nqf', 'sepsis_cdc', 'sepsis_cdc_simple']

top_features = [
    'urineoutput', 'lods', 'age', 'elixhauser_hospital', 
    'resprate_mean', 'lactate_mean', 'sofa', 
    'bun_mean', 'aniongap_max', 'sysbp_mean'
]


# --------------------------
# 2. Sidebar Filters
# --------------------------

st.sidebar.header("Filters")

# Select Sepsis Types
selected_sepsis = st.sidebar.multiselect(
    "Select Sepsis Types:",
    sepsis_types,
    default=["sepsis_angus"]  # Default pre-selected
)

# Select Clinical Features
selected_features = st.sidebar.multiselect(
    "Select Clinical Features:",
    top_features,
    default=["urineoutput", "age"]  # Default features
)

# Feature ranges
feature_ranges = {}
for feature in selected_features:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    feature_ranges[feature] = st.sidebar.slider(
        f"{feature} range:",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val)
    )

# --------------------------
# 3. Filter Data
# --------------------------

# Filter rows based on sepsis types
sepsis_filter = df[selected_sepsis].sum(axis=1) > 0
filtered_df = df[sepsis_filter]

# Apply filters for clinical features
for feature in selected_features:
    min_val, max_val = feature_ranges[feature]
    filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]

# Drop rows where top features have null values
filtered_df = filtered_df.dropna(subset=top_features)

# --------------------------
# 4. Survival Stats
# --------------------------

total_patients = len(filtered_df)
if total_patients > 0:
    # Count survived and died
    died_count = filtered_df["thirtyday_expire_flag"].sum()
    survived_count = total_patients - died_count

    # Percentages
    died_pct = (died_count / total_patients) * 100
    survived_pct = (survived_count / total_patients) * 100

    # Display results
    st.subheader("Survival Stats")
    st.write(f"**Total Patients Filtered**: {total_patients}")
    st.write(f"**Died**: {died_count} ({died_pct:.1f}%)")
    st.write(f"**Survived**: {survived_count} ({survived_pct:.1f}%)")
    

    # --------------------------
    # 5. Model Predictions
    # --------------------------

    # Scale data
    X = filtered_df[top_features].dropna(subset=top_features)  # Drop only for top features

    
    # Predict probabilities
    pred_probs = model.predict_proba(X)[:, 1]
    predicted_deaths = (pred_probs >= 0.5).sum()

    # Predicted percentages
    pred_death_pct = (predicted_deaths / total_patients) * 100

    st.subheader("Model Predictions")
    pred_probs = model.predict_proba(X)[:, 1]
    st.write("Average Probability:", pred_probs.mean())
    
else:
    st.warning("No patients match the selected criteria.")

# --------------------------
# 6. Boxplots for Clinical Features
# --------------------------

st.subheader("Boxplots for Clinical Features")

for feature in selected_features:
    fig = px.box(
        filtered_df,
        x="thirtyday_expire_flag",
        y=feature,
        color="thirtyday_expire_flag",
        title=f"{feature} by Survival Status (0 = Survived, 1 = Died)",
        labels={"thirtyday_expire_flag": "Thirty Day Expire Flag", feature: feature}
    )
    st.plotly_chart(fig)

# --------------------------
# 7. Optional: Show Filtered Data
# --------------------------

if st.checkbox("Show Filtered Data"):
    st.dataframe(filtered_df)
