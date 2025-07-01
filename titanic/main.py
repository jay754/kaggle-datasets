import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data
df = pd.read_csv('../train.csv')

# Step 1b: Explore data
print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nDescription:\n", df.describe())

# Step 2: Check for missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Step 3: Visualizations

# Survival count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Survival by sex
sns.countplot(x="Survived", hue="Sex", data=df)
plt.title("Survival by Sex")
plt.show()

# Survival by passenger class
sns.countplot(x="Survived", hue="Pclass", data=df)
plt.title("Survival by Class")
plt.show()

# Age distribution
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Step 4: Data Cleaning

# Drop columns that aren't useful for prediction
df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)

# Fill missing 'Age' with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\nData after cleaning:\n", df.head())
print("\nMissing values after cleaning:\n", df.isnull().sum())

# Step 5: Convert categorical variables to numbers

# Convert 'Sex' to 0/1 (male: 0, female: 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'])

# Step 6: Split into features (X) and label (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Step 7: Train/test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 8: Train logistic regression model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
