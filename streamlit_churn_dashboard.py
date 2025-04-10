import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Loading variables
@st.cache_data
def load_data():
    return pd.read_parquet("data/cleaned_churn_data.parquet")

df = load_data()

best_model = joblib.load("data/best_model.pkl")
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
fig = plt.figure(figsize=(10, 6))
sns.countplot(data=filtered_df, x="churn")
plt.xticks([0, 1], ["Not Churned", "Churned"])
st.pyplot(fig)

# Age Distribution
st.subheader("Age Distribution")
fig = plt.figure(figsize=(10, 6))
sns.histplot(filtered_df["age"], bins=20, kde=True)
st.pyplot(fig)

# SHAP feature importances for each customer churn prediction
st.subheader("SHAP Feature Importances for the selected customer churn prediction")
record_index = st.number_input("Select a customer index:", min_value=0, max_value=len(X_train_transformed)-1, step=1)

# Creates a placeholder for the SHAP feature importances plot
shap_feature_importance_placeholder = st.empty()

# Updates SHAP feature importances plot when customer index changes
@st.cache_data
def get_shap_values_for_record():
    explainer = shap.Explainer(best_model, X_test_transformed, feature_names=feature_names)
    shap_values = explainer(X_test_transformed)
    return shap_values

shap_values = get_shap_values_for_record()

fig = plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[record_index])
shap_feature_importance_placeholder.pyplot(fig, clear_figure=True)

# SHAP feature dependence plot
st.subheader("SHAP Feature Dependence Plot")
selected_feature = st.selectbox(
    "Select a feature to display in the dependence plot", 
    feature_names, 
    index=1
)

# Creates a placeholder for the SHAP global feature dependence plot
shap_global_feature_dependence_placeholder = st.empty()

# Converts X_train_transformed to a dataframe
X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=feature_names)

# Updates SHAP feature dependence plot when selected feature changes
@st.cache_data
def get_shap_values_for_feature_dependence():
    explainer_train = shap.Explainer(best_model, X_train_transformed_df)
    shap_values_train = explainer_train(X_train_transformed_df)
    return shap_values_train

shap_values_train = get_shap_values_for_feature_dependence()

fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot(
    selected_feature, 
    shap_values_train.values, 
    X_train_transformed_df, 
    ax=ax
)
shap_global_feature_dependence_placeholder.pyplot(fig, clear_figure=True)
