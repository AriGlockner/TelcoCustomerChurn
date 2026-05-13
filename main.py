import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('data/telco_customer_churn.csv')
df.drop(['customerID'], axis=1, inplace=True)

# Prepare features for ML
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Your existing mappings
df['gender'] = df['gender'].map({'Male':1, 'Female':0})
df['InternetService'] = df['InternetService'].map({'No':0, 'DSL':1, 'Fiber optic':2})
df['Contract'] = df['Contract'].map({'Month-to-month':0, 'One year':1, 'Two year':2})
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

df.to_csv('data/prepared_data.csv', index=False)

# Split features into train and test features
y = df['Churn']
X = df.drop(['Churn'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
