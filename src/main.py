import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

random_state = 42
max_iter = 1000
nearest_neighbors = 5
hidden_layer_sizes = (100,)

df = pd.read_csv('../data/prepared_data.csv')
# Split features into train and test features
y = df['Churn']
X = df.drop(['Churn'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

models = [
    RandomForestClassifier(random_state=random_state),
    GradientBoostingClassifier(random_state=random_state),
    HistGradientBoostingClassifier(random_state=random_state),
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, random_state=random_state)),
    make_pipeline(StandardScaler(), RidgeClassifier()),
    make_pipeline(StandardScaler(), SVC()),
    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=nearest_neighbors)),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
]

model_labels = [
    'RandomForest', 'GradientBoosting', 'HistGradientBoosting',
    'LogisticRegression', 'Ridge', 'SVC', 'KNN', 'GaussianNB', 'LDA', 'MLP'
]
scores = []
highest_accuracy = -1
best_model = None

for model, label in zip(models, model_labels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    scores.append(score)

    if score > highest_accuracy:
        highest_accuracy = score
        best_model = model

# Print the best model
print(f'The {best_model} model had the highest accuracy of {highest_accuracy}')

# Creat a plot of the models
plt.bar(model_labels, scores)
plt.ylim(0.76, 0.83)
plt.xticks(model_labels)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation='vertical')
plt.xlabel('Model')
plt.tight_layout()

# Save the output from the model
plt.savefig('figures/baseline_model_comparison.png')
plt.show()

# Sort and print each model from most to least accurate
results = zip(model_labels, scores)

results = sorted(results, key=lambda x: x[1], reverse=True)
for label, score in results:
    print(f'{label}: {score}')

