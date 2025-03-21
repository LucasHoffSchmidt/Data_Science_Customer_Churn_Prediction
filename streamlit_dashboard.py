import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import PartialDependenceDisplay

# Loading variables
@st.cache_data
def load_data():
    return pd.read_parquet("data/cleaned_churn_data.parquet")

df = load_data()

best_model = joblib.load("data/best_model.pkl")
best_preprocessor = joblib.load("data/best_preprocessor.pkl")
X_train_transformed = joblib.load("data/X_train_transformed.pkl")
X_test_transformed = joblib.load("data/X_test_transformed.pkl")
feature_names = joblib.load("data/feature_names.pkl")

# App title
st.title("Telecom Churn Data Interactive Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
selected_partner = st.sidebar.multiselect("Telecom Partner", df["telecom_partner"].unique(), 
                                          default=df["telecom_partner"].unique())
selected_gender = st.sidebar.multiselect("Gender", df["gender"].unique(), default=df["gender"].unique())
selected_state = st.sidebar.multiselect("State", df["state"].unique(), default=df["state"].unique())
selected_month = st.sidebar.multiselect("Month", df["month"].unique(), default=df["month"].unique())

# Filter dataset
filtered_df = df[
    (df["telecom_partner"].isin(selected_partner)) &
    (df["gender"].isin(selected_gender)) &
    (df["state"].isin(selected_state)) &
    (df["month"].isin(selected_month))
]

# Show filtered dataset
st.subheader("Filtered Data")
st.write(filtered_df.head())

# Churn Distribution
st.subheader("Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x="churn", data=filtered_df, ax=ax)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Not Churned", "Churned"])
st.pyplot(fig)

# Age Distribution
st.subheader("Age Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df["age"], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# SHAP Bar Plot for Single Record
st.subheader("SHAP Feature Contributions for Selected Record")
record_index = st.number_input("Select a record index:", min_value=0, max_value=len(X_train_transformed)-1, step=1)
fig = plt.figure(figsize=(10, 6))
explainer = shap.Explainer(best_model, X_test_transformed, feature_names=feature_names)
shap_values = explainer(X_test_transformed)
shap.plots.bar(shap_values[record_index])
st.pyplot(fig)

# Partial Dependence Plots
st.subheader("Partial Dependence Plots")
selected_features = st.multiselect(
    "Select variables to display in the Partial Dependence Plots", 
    feature_names, 
    default=[feature_names[0], feature_names[1]]
)

fig, ax = plt.subplots(figsize=(10, 6))
disp = PartialDependenceDisplay.from_estimator(
    best_model, 
    X_train_transformed, 
    features=selected_features, 
    feature_names=feature_names, 
    n_cols=2, 
    ax=ax
)

plt.subplots_adjust(hspace=0.5)
st.pyplot(fig)