# Data Science: Customer Churn Prediction

This project aims to identify customers at risk of churning and provide insights into how we can reduce churn rates.

## 📊 Business Problem

We are an **Indian telecom company** experiencing customer churn. Our goal is to understand why customers are leaving and determine actionable steps to reduce churn.

## 📂 Data Collection

The data comes from telecom churn records of Indian telecom companies. The dataset consists of **243,553 rows**, of which we sample **10,000 rows** to optimize time spent training models.

### Attributes in the dataset:
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
- The state of **Sikkim** and the city of **Kolkata** had the highest churn rates.

---

## 🔧 Data Preprocessing

In the preprocessing stage, we check for missing and invalid values, remove duplicates and create new features. The following changes were applied:

1. **date_of_registration** converted from **object** to **datetime**.
2. Removed any leading or trailing spaces from categorical features.
3. Created new **binned features**:
   - **age_bracket**: Categorical variable (young, middle-aged, old).
   - **salary_bracket**: Categorical variable (low_salary, mid_salary, high_salary).
   - **month**: The month (1-12) when the customer registered with the telecom provider.
4. **Visualization** of churn rate for the new variables reveals:
   - **Age bracket** has minimal impact on churn.
   - **Salary bracket** has moderate impact on churn.
   - **Month** has a significant impact on churn.
5. We drop variables with **high cardinality** (many unique values) and those with **low variance** between churn and no churn. 

---

## 🧠 Model Training and Evaluation

We apply machine learning to train models using the following features:
- **telecom_partner**
- **state**
- **city**
- **num_dependents**
- **month**
- **age**
- **salary_bracket**

### Training Process:
1. Split the dataset into **training (80%)** and **testing (20%)** sets, ensuring **stratification** to handle class imbalance.
2. **One-Hot Encoding** applied to categorical features.
3. **SMOTE** (Synthetic Minority Over-sampling Technique) used to oversample churned customers to ensure balance between values.
4. **Pipelines** created to go through the process of applying one-hot encoding, SMOTE, and training the classifier model.
5. **Hyperparameter tuning** performed using **GridSearchCV** based on **F1 score**, balancing precision and recall.
6. Evaluated each model, selecting the one with the highest f1-score and at least **70% accuracy**.

---

## 🧐 Model Interpretation

We interpret the model using:
- **Feature Importances** from the best-performing model to explain which factors has the highest influence in predicting churn.
- **SHAP (Shapley Additive Explanations) summary plot** to explain the global impact of features on churn prediction.
- **SHAP mean values** to see whether features are more likely to cause churn or non-churn on average.
- **Count of positive and negative SHAP values per feature** to see the distribution of positive(churn likely) and negative(churn unlikely) shap values for each feature.
- **Precision-Recall Curve** To assess the model's ability to distinguish churners from non-churners.
- **Partial Dependence Plots (PDP)** to examine the isolated impact of selected features on churn.

### Findings:
- The state of Chennai seems to strongly induce non-churning. It has both the highest model feature importance at 0.05 and the lowest mean shap value of -0.12.
- Age fluctuates a lot and may both contribute positively and negatively to churning. On average it induces churn, with higher churn rates at about 20, 40 and 65 years old.
- The state of Assam seems to moderately cause churn with the highest mean shap value and a partial dependence value of more than 0.25.
- Number of children (num_dependents) seem to strongly induce non-churning at 1 and 4 children, having an average SHAP value of -0.11 and -0.07 respectively.
- The first month also seems to consistently be related with low churning, with high importance in both feature importance, shap summary plot, mean shap value at -0.07 and a high number of negative shap values at 1573.

---

## 🚀 Model Deployment

We deploy the model via a **Streamlit dashboard** with interactivity and dynamic visuals. Key features include:

- A **title** for the dashboard.
- **Sidebar filters** to change the variables and update visuals dynamically.
- A **filtered dataset** that reflects the changes based on the selected filters.
- A **countplot** showing churn distribution.
- A **histplot with KDE** for age distribution.
- A **SHAP individual feature contribution plot** to see feature contributions to churn prediction for the selected customer.
- **Partial dependence plots** for the selected variables.

We save model components in the **.pkl** format and the dataset in the **.parquet** format for loading into the Streamlit app.

The streamlit app can be accessed here: [Customer churn streamlit app](https://data-science-customer-churn-prediction.streamlit.app/)

---

## Conclusion
- We should experiment with marketing campaigns targeting people around the age of 40, as they exhibit high purchasing power and may be more receptive to retention efforts. 
- We should also consider advertising directly to people living in Assam, where churn rates appear to be higher than in other states. 
- Furthermore people with no children may churn more often due to different financial priorities and lifestyle flexibility, so offering special discounts or tailored incentives for them, could improve retention. 
- Lastly data suggests that new subscriptions peak in the beginning of a new year, so strategic promotions during the December holiday season could help attract and retain customers.   

By conducting targeted experiments based on these insights, we should be able to refine our understanding of churn drivers and develop more effective customer retention strategies. 
