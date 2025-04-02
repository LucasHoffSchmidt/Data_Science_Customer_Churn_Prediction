# Data Science: Customer Churn Prediction
This project seeks to identify customers at risk of churning and provide insights into how we can reduce churn rates.

## 📊 Business Problem
We are an Indian telecom company selling phone subscriptions to customers. We have seen a rise in the amount of customers who are churning and would like to know why this might be the case and what we can do about it.

## 📂 Data Collection
### Objective
To find relevant telecom customer data that includes whether or not the customer has churned. 

### Process
- Searched for telecom datasets.
- Found a kaggle dataset containing telecom churn records of the Indian telecom companies Airtel, Reliance Jio, Vodafone and BSNL.
- The dataset consists of a whopping **243,553 rows**, so we sample **10,000 rows** to optimize time spent training models.
- Got an overview of the different attributes included in the dataset:
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

### Findings
- We notice that calls_made has a negative value of -2, which is not a realistic number. The lowest number of calls should be 0, indicating that the customer has made no calls at all. 

---

## 🔍 Data Understanding: Exploratory Data Analysis (EDA)
### Objective
Identify and fix obvious data errors such as incorrect data types and missing values and visualize the features' relationship with the target variable churn.  

### Process
- Checked for null values and wrong data types.
- Changed the data type of date_of_registration to datetime.
- Checked the **statistical distribution** of attributes and notice negative values for the features calls_made, sms_sent and data_used.
- Limited the lower bound of features with negative values to 0.
- Used matplotlib and seaborn to **visualize features** relationship with the target variable using histograms, heatmaps, count plots, bar plots, box plots and line plots.

### Findings
- The dataset has no null values.
- About 2-3% of calls_made, sms_sent and data_used was negative, which may mean that many customers use these so little, that their budgets overflow to the next month. We reduce these to a lower bound of 0, since they cannot be negative in reality.  
- All features seem to only have a **minor correlation with churn**, with the highest correlations being with pincode, estimated_salary and calls_made at about 0.01 correlation.
- We see that on average 20% of people churn and 80% do not churn.
- Women and customers with 0 children are more likely to churn.
- The telecom partner Vodafone has the highest churn rate at a little above average at about 21%.
- The state of Sikkim has the highest average churn rate of all states at 25%.
- The age of customers least likely to churn is 51 with a churn rate of about 13%, and the most likely to churn is 22 with a churn rate of about 26%. A 22 year old is therefore twice as likely to churn as a 51 year old. 
- The city of Kolkata has the highest churn rate at about 21%.

---

## 🔧 Data Preprocessing
### Objective
Handle missing and invalid values, remove any duplicates or outliers found during EDA and ensure we have the necessary features needed for machine learning. 

### Process
- Removed any duplicate rows
- Checked that we have no ages less than 0.
- Checked for invalid churn values.
- Removed any leading and trailing spaces from categorical features.
- Created aggregated features:
   - **Data used bracket**: Aggregated data_used feature to data used bracket with low, mid and high data used.
   - **Salary bracket**: Aggregated estimated_salary feature to salary bracket with low, mid and high estimated salary.
   - **Month**: Aggregated date_of_registration to month with numbers for each month of the year.
- Dropped any high cardinality feature with no real predictive signal potential. 

### Findings
- There are no duplicates, missing or invalid values in the dataset
- We keep the aggregated features salary, data used and month, since they all show clear variations. 
- We drop any high cardinality features to ensure machine learning models generalize, rather than memorize, reducing the likelihood of overfitting.

---

## 🧠 Model Training and Evaluation
### Objective
Train classifier machine learning models to predict which customers will churn and evaluate the models to determine the best. 

### Process
- Splitted the data into training and testing data, dropping the salary_bracket which greatly reduced evaluation metrics.
- Used **ordinal and onehot encoding** on nominal and ordinal categorical features.
- Created a pipeline to automatically apply preprocessing to features.
- Created a **hyperparameter grid** to test different model configurations.
- Made a **grid search** cross validation on all models and hyperparameter grids using stratified k-fold to ensure consistent class proportions in each tested data fold.
- Evaluated each model, selecting the best based on the greatest f1-score and an accuracy of at least 70% overall. 

### Findings
We determine that the best model is XGBoost with a recall of 0.1910, an f1_score of 0.2141 and an accuracy of 0.7210. 

---

## 🧐 Model Interpretation
### Objective
To explain how the model makes predictions, by analyzing the impact of features. 

### Process
- Made a **feature importances plot** that shows the feature's impact on the model's decision making process.
- Made a **SHAP summary plot** that reveals how features impacted individual predictions based on their value.
- Made a **mean SHAP values plot** that shows the overall mean SHAP value of a feature, indicated whether it on average contributed mostly to increasing or decreasing the likelihood of churning.
- Made a **receiver operating characteristic (ROC) curve**, that determines the models ability to distinguish churning from non-churning.
- Made **partial dependence plots** of features with highest impact according to the SHAP summary plot.

### Findings
- **States** have the highest feature impact, noticeably the state of Himachal Pradesh, which contributes 3% to the model's overall decisionmaking process.
- **Gender** has the lowest feature importance.
- Higher numbers of **calls and sms' sent** generally increases likelihood of not churning.
- **Later months** increases the likelihood of churn, while **earlier months** reduce it.
- **High data usage** is linked to churning.
- Customers with more **children and females** increase the likelihood of churning. 
- The model performs just slightly better than random guessing at distinguishing churning from non-churning, presumably due to low correlation between features and churn with the highest numerical correlation being 0.01.

---

## 🚀 Model Deployment
### Objective
To create a streamlit interactive dashboard with relevant visualizations. 

### Process
- We create an interactive streamlit dashboard with the following features:
   - A **title** for the dashboard.
   - **Sidebar filters** to change the feature variables and update visuals dynamically.
   - A **filtered dataset** that reflects the changes based on the selected filters.
   - A **countplot** showing churn distribution.
   - A **histplot with KDE** showing age distribution.
   - A **SHAP individual feature contribution plot** to see feature contributions to churn prediction for the selected customer.
   - A **SHAP dependence plot** for the selected feature.
- We save model components in the **.pkl** format and the dataset in the performant **.parquet** format for loading into the Streamlit app.

The streamlit app can be accessed here: [Customer churn streamlit app](https://data-science-customer-churn-prediction.streamlit.app/)

### Findings
 - The launched model performs well with cached data that enables one time loading and subsequent real-time interactive analysis. 

---

## Conclusion
- **Higher churn rates are linked with:**
   - Minimal engagement with a low number of calls and sms' sent. 
   - Months later in the year.
   - High data usage.
   - Customers with children.
   - Women.
- **To drive retention we should therefore:**
   - Look out for customers with low engagement, and offer them tailored deals such as bonuses or a loyalty program that allows them to earn spending points by calling and texting.
   - Create holidaythemed competitions later in the year, to stimulate and engage customers.
   - Create data heavy subscription packages with a lesser focus on calls and sms' sent.
   - Provide family packages to incentivize families to keep their subscription, as well as sign up their children for one.
   - Create a feedback function that allow people to voice wishes and concerns, to reveal the most important drivers for women, and subsequently implementing these. 
