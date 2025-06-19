# SUGAR: Smart Utilization of Glucose & AI for Risk Prediction
A machine learning-powered tool designed for robust diabetes classification from patient datasets, aiming for web-server deployment.

The input dataset - https://www.kaggle.com/code/kashafabbas036/diabetes-dataset-preprocessing


**The Challenge: Precision in Diabetes Classification**
Accurate and early classification of diabetes is crucial for timely intervention and improved patient outcomes. Existing methods often struggle with optimal predictive performance and accessibility for non-technical users.

**Our Solution: SUGAR**
This project introduces SUGAR (Smart Utilization of Glucose & AI for Risk prediction), a machine learning tool built to enhance the precision of diabetes classification. Leveraging the comprehensive Kaggle Diabetes Prediction Dataset, SUGAR develops and evaluates advanced predictive models. Our primary goal is to identify the most effective classification model, with the ultimate aim of deploying it via a user-friendly web server to make accurate diabetes risk prediction readily accessible.

### Initial data exploration
The first step is to explore the dataset to understand its structure, identify key features, and uncover initial patterns or anomalies.

![Binning of Age into groups and diabetes comparison across the gender](images/diabetes_prevelance_age_group.png)

Followed by looking into distribution of blood glucose level across genders
![Distribution of blood glucose level](images/distribution_blood_glucose.png)



Based on the dataset, further explored the key features that have an imapct in diabetes prediction
![Heatmap of features impacting diabetes](images/heatmap_diabetes_features.png)

In the above heatmap, HbA1c_level (40%) and blood_glucose_level (42%) exhibit a significant positive correlation with diabetes. This strong relationship is medically consistent: higher values in these crucial blood markers directly indicate a greater probability of diabetes.