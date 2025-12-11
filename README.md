# AlphaCare Insurance Solutions – Week 3 Analytics Project

## Project Overview
This project focuses on building predictive models for **motor insurance claims** in South Africa using historical insurance data. The goal is to provide ACIS with actionable insights for **risk management, marketing, and premium pricing** strategies.  

We develop models to:  
1. Predict **claim probability** (classification).  
2. Predict **claim severity** (regression).  
3. Analyze **feature importance** to inform data-driven business decisions.  

---

## Dataset
The dataset contains 50+ features for each policy, including:  
- Policy metadata: `PolicyID`, `TransactionMonth`, `Bank`, `AccountType`, `MaritalStatus`, etc.  
- Vehicle data: `make`, `Model`, `RegistrationYear`, `VehicleType`, `VehicleAge`, `Cylinders`, `Kilowatts`, etc.  
- Insurance coverage details: `CoverType`, `CoverGroup`, `TotalPremium`, `SumInsured`, etc.  
- Claim information: `claim_indicator`, `TotalClaims`.  

---

## Analytical Approach

### 1. Data Preprocessing
- Handle missing values, outliers, and inconsistent data.  
- Encode categorical variables using **OneHotEncoder** or **Target Encoding**.  
- Scale numeric features as required.  

### 2. Feature Engineering
- Derive new features such as `VehicleAge`, `MonthlyPremiumChange`, `ClaimFrequencyPerMonth`.  
- Keep all features relevant for classification and regression tasks.  

### 3. Exploratory Data Analysis (EDA)
Key visualizations include:  
1. Correlation matrix of numeric features  
2. Premium change vs. claim change  
3. Top 10 postal codes by claim frequency  
4. Total claims vs. claims by cover type  
5. Monthly premium trends for top postal codes  
6. Loss ratio by province  
7. Loss ratio by vehicle type  
8. Loss ratio by gender  
9. Monthly claim frequency trend  
10. Monthly claim severity trend for top vehicle models  

> Placeholders are included for visualizations 1–10.  

### 4. Modeling
**Claim Probability (Classification)**  
- Models used: Logistic Regression, Random Forest, XGBoost  
- Train-test split: 80:20  
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC  

**Claim Severity (Regression)**  
- Models used: Linear Regression, Random Forest Regressor, XGBoost Regressor  
- Train-test split: 80:20  
- Evaluation metrics: RMSE, MAE, R²  

### 5. Model Interpretability
- **SHAP** (SHapley Additive exPlanations) used for global and local interpretability.  
- Feature importance analysis identifies the most influential factors affecting claims and premiums.  

---

## Key Findings
- **XGBoost** outperforms other models in both classification and regression tasks.  
- Top features impacting claims include `VehicleAge`, `TotalPremium`, `PostalCode`, `Vehicle Make/Model`, and `CoverType`.  
- Certain postal codes and older vehicles significantly increase the likelihood and severity of claims.  
- Gender and vehicle type also influence loss ratios, highlighting opportunities for targeted marketing and premium adjustments.  

---

## Recommendations
- Use **XGBoost models** for both claim probability and severity predictions.  
- Adjust premiums based on `VehicleAge`, `PostalCode`, and `CoverType`.  
- Target marketing campaigns to low-risk customers identified by the claim probability model.  
- Monitor high-risk postal codes for potential fraud or risk mitigation.  

---

## Future Work
- Perform **A/B hypothesis testing** for marketing campaigns and promotional strategies.  
- Expand models to include additional time-series trends and seasonal effects.  
- Deploy interactive dashboards for real-time claim prediction and monitoring.  

---

## Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
