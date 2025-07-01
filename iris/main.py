import pandas as pd
from sklearn.datasets import load_iris

# Step 1: Load dataset
iris = load_iris()

# Convert to pandas DataFrame for easier exploration
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print(df.head())

# Step 2: Explore the Data

print(df.info())
print(df.describe())
print(df['species'].value_counts())

# Step 3: Visualize

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df, hue='species')
plt.show()

# Step 4: Prepare Data for ML

from sklearn.model_selection import train_test_split

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train a Classifier

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 6: Evaluate

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
