import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Global constraints
random_state = 42
max_iter = 1000
num_nearest_neighbors = 5
hidden_layer_sizes = (100,)

readme_filepath = '../README.md'

df = pd.read_csv('../data/prepared_data.csv')

# Split features into train and test features
y = df['Churn']
X = df.drop(['Churn'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Store each model to test in a list to iterate through each model for the baseline test
models = [
    RandomForestClassifier(random_state=random_state),
    GradientBoostingClassifier(random_state=random_state),
    HistGradientBoostingClassifier(random_state=random_state),
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, random_state=random_state)),
    make_pipeline(StandardScaler(), RidgeClassifier(random_state=random_state)),
    make_pipeline(StandardScaler(), SVC(random_state=random_state)),
    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=num_nearest_neighbors)),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
]

# Store the names of each model
model_labels = [
    'RandomForest', 'GradientBoosting', 'HistGradientBoosting',
    'LogisticRegression', 'Ridge', 'SVC', 'KNN', 'GaussianNB', 'LDA', 'MLP'
]

# Store the results
scores = []

# Train and Test each model
for model, label in zip(models, model_labels):
    model.fit(X_train, y_train)
    scores.append(f1_score(y_test, model.predict(X_test)))

'''# Creat a bar graph of the models
plt.bar(model_labels, scores)
plt.ylim(0.76, 0.83)
plt.xticks(model_labels)
plt.ylabel('Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation='vertical')
plt.xlabel('Model')
plt.tight_layout()

# Save the results from the bar graph
plt.savefig('../figures/baseline_model_comparison.png')
plt.show()'''

# Sort and print out the results
print('\nModel F1 Scores:')

results = sorted(zip(model_labels, scores), key=lambda x: x[1], reverse=True)
for label, score in results:
    print(f'{label}: {score}')


df = pd.DataFrame(results, columns=['Model', 'Score'], index=range(1, len(results) + 1))
table_md = df.to_markdown(index=False)

with open(readme_filepath, 'r') as f:
    content = f.read()

with open(readme_filepath, 'w') as f:
    f.write(content.replace('<!-- INSERT_BASELINE_TABLE -->', table_md))
