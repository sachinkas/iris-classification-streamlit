import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Iris Dataset
df = pd.read_csv('Iris.csv')
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test data
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.pkl')

# Train 5 Classifiers
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"model/{name}.pkl")

print("Training complete! Models and test_data.csv saved successfully.")