import pandas as pd

def clean_data(input_filepath='../data/telco_customer_churn.csv', output_filepath='../data/prepared_data.csv'):
    # Prep the data
    df = pd.read_csv(input_filepath)
    df.drop(['customerID'], axis=1, inplace=True)

    # Prepare features for ML
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Your existing mappings
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
    df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Mailed check': 0,
        'Electronic check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })

    # Binary features
    binary_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'PaperlessBilling', 'Churn']

    for feature in binary_features:
        df[feature] = df[feature].map({
            'Yes': 1,
            'No': 0,
            'No phone service': 0,
            'No internet service': 0
        })

    df.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    clean_data()
