# 🏦 Bank GoodCredit — Credit Risk Scoring

## Overview
This project predicts whether a bank's credit card customer 
will default on payments (Bad_label), using customer demographic, 
account history, and credit enquiry data.

## Problem Statement
Bank GoodCredit wants to predict a credit risk score for current 
credit card customers, denoting creditworthiness, to help reduce 
default risk.

## Dataset
Data was extracted from a live MySQL database with 3 related 
tables (Cust_Demographics, Cust_Account, Cust_Enquiry), 
containing 23,896 customers with 79 anonymized demographic 
features plus account and enquiry history.

Due to credentials and live-access requirements, the database 
is not included in this repository. Connection code has been 
redacted - replace placeholder host/username/password to run.

- Target: Bad_label (0 = Good credit, 1 = Bad credit / 30+ DPD)
- Severe class imbalance: 95.8% Good vs 4.2% Bad

## Approach
- Diagnosed and resolved hidden missing data masked as empty strings
- Engineered features by parsing raw payment-history strings into 
  days-past-due buckets and enquiry recency windows
- Applied SMOTE and One-Hot Encoding for categorical features
- Aggregated multi-table account and enquiry data to customer level
- Trained and compared Logistic Regression, Random Forest, and XGBoost

## Results
| Model | Gini Coefficient |
|---|---|
| Logistic Regression | 0.2992 |
| Random Forest | 0.2760 |
| XGBoost | 0.2525 |

**Best Model:** Logistic Regression — selected based on Gini 
coefficient and rank-ordering performance.

## Key Insight
Despite engineering domain-suggested features (DPD buckets, 
enquiry recency), demographic features such as employment type 
and education level were stronger predictors than transactional 
payment behavior in this dataset.

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, 
Imbalanced-learn (SMOTE), MySQL Connector
