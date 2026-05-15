# TelcoCustomerChurn
## 1) Introduction

This project analyzes the Telco Customer Churn dataset, a fictional dataset representing 7,043 telecommunications customers in California during Q3. The dataset contains customer attributes and service usage patterns with the goal of predicting customer churn - identifying which customers are likely to leave so the business can develop targeted retention programs.

The dataset includes 18 features including:
- Customer demographics (gender, SeniorCitizen, Partner, Dependents, tenure)
- Service Subscriptions (Phone Service, Multiple Lines, Internet Service, Online Security, Online Backup, Device Protection, Technical Support, Streaming TV, and Streaming Movies)
- Accounts & Billing (Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges)

Target Variable: Churn (Yes/No)

Objective: Compare multiple machine learning classifiers to establish a performance baseline, then optimize top-performing algorithms through hyperparameter tuning to develop a final production-ready model for predicting customer churn.

## 2) Data Preprocessing
The raw Telco Customer Churn dataset contains mixed data types requiring preprocessing before model training. The following transformations were applied:

| Step                     | Transformation                                                                                                                                                         | Details                                                                                  |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Drop identifier**      | Remove `customerID`                                                                                                                                                    | Unique per customer; no predictive value                                                 |
| **Target encoding**      | `Churn`: Yes → 1, No → 0                                                                                                                                               | Binary classification requires numeric labels                                            |
| **Binary encoding**      | `gender`: Female → 0, Male → 1; `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`: Yes/No → 1/0                                                              | Simple binary categorical features                                                       |
| **Multi-class encoding** | `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod` | Converted to integer codes (0, 1, 2, ...) preserving categorical information             |
| **Numeric conversion**   | `TotalCharges`: string → float                                                                                                                                         | Original format stored as object due to empty strings; coerced errors to NaN and imputed |
| **Feature scaling**      | `StandardScaler` applied to `tenure`, `MonthlyCharges`, `TotalCharges`                                                                                                 | Required for distance-based algorithms (SVC, LogisticRegression, KNN, Ridge) .org        |

## 3) Baseline Model Comparison

10 classifiers from scikit-learn were evaluated using default parameters with standardized preprocessing where appropriate. The F1 score metric was used for model comparison due to the imbalanced nature of the dataset. The results are summarized in the table below:

|    | Model                |    Score |
|---:|:---------------------|---------:|
|  0 | GaussianNB           | 0.648588 |
|  1 | LDA                  | 0.628986 |
|  2 | LogisticRegression   | 0.626996 |
|  3 | Ridge                | 0.603369 |
|  4 | GradientBoosting     | 0.599106 |
|  5 | HistGradientBoosting | 0.584071 |
|  6 | SVC                  | 0.580343 |
|  7 | MLP                  | 0.568421 |
|  8 | RandomForest         | 0.560250 |
|  9 | KNN                  | 0.544681 |

Table directly generated from the src/generate_baselines.py script

### TODO: Key Findings
| - | Finding                          | Interpretation                                                                                                                                                                                                    |
|---|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | Top Performers                   | GaussianNB, LDA, Logistic Regression, and Ridge Regression were the top-performing models with F1 scores above 0.60. These linear models outperformed tree-based models like Gradient Boosting and Random Forest. |

The Telco dataset's preprocessing produced mostly binary features and a few one hot encoded categorical features, which may explain why linear models performed better than tree-based models. The lack of strong non-linear relationships and the presence of many binary features likely favored algorithms that can effectively model linear decision boundaries.

## 4) Hyperparameter Optimization

| - | **Model**           | **Best Parameters**                                                                  | Baseline Accuracy | Optimized Accuracy | Notes                                                               |
|---|---------------------|--------------------------------------------------------------------------------------|-------------------|--------------------|---------------------------------------------------------------------|
| 1 | LDA                 | {'solver': 'svd'}                                                                    | 0.8183            | 0.8183             | No improvement; default parameters already optimal for this dataset |
| 2 | Logistic Regression | {'C': 0.01, 'class_weight': None, 'l1_ratio': 0, 'solver': 'saga'}                   | 0.8176            | 0.8204             | -                                                                   |
| 3 | Gradient Boosting   | {'learning_rate': 0.05, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 200} | 0.8091            | 0.8062             | -                                                                   |
