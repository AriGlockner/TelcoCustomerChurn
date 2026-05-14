import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Global constraints
random_state = 42
max_iterations = 5000

# Processed data filepath
df = pd.read_csv('../data/prepared_data.csv')

# Split features into train and test features
y = df['Churn']
X = df.drop(['Churn'], axis=1)

model_labels = ['LDA', 'LogisticRegression', 'GradientBoosting']

models = [
    LinearDiscriminantAnalysis(),
    LogisticRegression(max_iter=max_iterations, random_state=random_state),
    GradientBoostingClassifier(random_state=random_state),
]

param_grids = [
    {'solver': ['svd', 'lsqr', 'eigen']},  # LDA

    {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': [0, 0.5, 1],  # 0=L2, 1=L1, 0.5=elastic net
        'solver': ['saga'],  # saga supports all l1_ratio values
        'class_weight': [None, 'balanced']
    },  # LogisticRegression

    {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10]
    }  # GradientBoosting
]

for model, label, param_grid in zip(models, model_labels, param_grids):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    print(f'{label}: {grid.best_params_}')
