# Data Science: Customer Churn Prediction

This project aims to identify customers at risk of churning and provide insights into how we can reduce churn rates.

## 📊 Business Problem

We are an **Indian telecom company** experiencing customer churn. Our goal is to understand why customers are leaving and determine actionable steps to reduce churn.

## 📂 Data Collection

The data comes from telecom churn records of Indian telecom companies. The dataset consists of **243,553 rows**, but for the purpose of reducing processing time, we sample **10,000 rows**.

### Variables in the dataset:
- **customer_id**: Unique identifier for each customer.
- **telecom_partner**: Telecom provider for the customer.
- **gender**: Gender of the customer.
- **age**: Age of the customer.
- **state**: State in India where the customer resides.
- **city**: The city where the customer is located.
- **pincode**: Customer's location pincode.
- **date_of_registration**: The registration date with the telecom provider.
- **num_dependents**: Number of dependents (e.g., children).
- **estimated_salary**: Customer's estimated salary.
- **calls_made**: Number of calls made by the customer.
- **sms_sent**: Number of SMS messages sent.
- **data_used**: Amount of data used by the customer.
- **churn**: Binary variable indicating if the customer churned (1 = churned, 0 = not churned).

---

## 🔍 Data Understanding: Exploratory Data Analysis (EDA)

During the EDA stage, we check for missing data, data types, and statistical distributions. 
We also visualize the distribution of features and their relationship with the target variable **churn**, using **matplotlib** and **seaborn**.

### Key Insights:
- **No missing values** in the dataset.
- **date_of_registration** is an object type but should be a datetime type.
- Approximately **80%** of customers that registered between **2020-01-01 and 2020-05-03** did **not churn**, while about **20%** did.
- Churning seems to be **independent** of variables like estimated salary, SMS sent, data used, and number of dependents.
- **Women are more likely to churn** than men.
- **Vodafone** had the highest churn rate.
- **Age** exhibits fluctuating churn rates, with lower churn around age **50**.
- **Sikkim** (state) and **Kolkata** (city) had the highest churn rates.

---

## 🔧 Data Preprocessing

In the preprocessing stage, we check for missing and invalid values and remove duplicates. The following changes were applied:

1. **date_of_registration** converted from **object** to **datetime**.
2. Removed any leading or trailing spaces from string variables.
3. Created new **binned variables**:
   - **age_bracket**: Categorical variable (young, middle-aged, old).
   - **salary_bracket**: Categorical variable (low_salary, mid_salary, high_salary).
   - **month**: The month (1-12) when the customer registered.
4. **Visualization** of churn rate for the new variables reveals:
   - **Age bracket** has minimal impact on churn.
   - **Salary bracket** has moderate impact on churn.
   - **Month** has a significant impact on churn.
5. We drop variables with **high cardinality** (many unique values) and those with **low variance** between churned and non-churned customers.

---

## 🧠 Model Training and Evaluation

We apply machine learning to train models using the following features:
- **telecom_partner**
- **gender**
- **state**
- **city**
- **month**
- **age_bracket**
- **salary_bracket**

### Training Process:
1. Split the dataset into **training (80%)** and **testing (20%)** sets, ensuring **stratification** to handle class imbalance.
2. **One-Hot Encoding** applied to categorical features.
3. **SMOTE** (Synthetic Minority Over-sampling Technique) used to oversample churned customers (value = 1) for balance.
4. **Pipelines** created to include one-hot encoding, SMOTE, and classifier model with **balanced class weights**.
5. **Hyperparameter tuning** performed using **GridSearchCV** based on **F1 score**, balancing precision and recall.
6. Evaluated each model, selecting the one with the highest recall and at least **70% accuracy**.

---

## 🧐 Model Interpretation

We interpret the model using:
- **Feature Importances** from the best-performing model.
- **SHAP (Shapley Additive Explanations)** to explain the global impact of features on churn prediction.
- **Partial Dependence Plots (PDP)** to examine the impact of selected features.

### Findings:
- **Madhya Pradesh** state, and **Months 1, 5, and 11** show high correlation with churn.
- **Gender** has a **negligible** impact on churn.

---

## 🚀 Model Deployment

The model is deployed via a **Streamlit dashboard**. Key features include:

- A **title** for the dashboard.
- **Sidebar filters** to change the variables and update visuals dynamically.
- A **filtered dataset** that reflects the changes based on the selected filters.
- A **countplot** showing churn distribution.
- A **histplot with KDE** for age distribution.
- **Partial dependence plots** for the selected variables.

We save model components in the **.pkl** format and the dataset in the **.parquet** format for loading into the Streamlit app.

---

## Conclusion
We should experiment with marketing campaigns targeting women, people around the age of 50, people living in Madhya Pradesh and applying marketing campaigns in the 1st, 5th and 11th month of the year. 
Based on our findings in these experiments, we should be able to narrow down the cause of the increasing churn rates. 
