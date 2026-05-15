import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Global constraints
random_state = 42
df = pd.read_csv('../data/prepared_data.csv')

# Split features into train and test features
y = df['Churn']
X = df.drop(['Churn'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

def run(models, model_labels, print_output=True):
    # Store the results
    scores = []

    # Train and Test each model
    for model, label in zip(models, model_labels):
        model.fit(X_train, y_train)
        scores.append(f1_score(y_test, model.predict(X_test)))

    # Sort the results
    results = sorted(zip(model_labels, scores), key=lambda x: x[1], reverse=True)

    # Print the output
    if print_output:
        print('\nModel F1 Scores:')
        for label, score in results:
            print(f'{label}: {score}')

    return results
