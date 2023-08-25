import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

titanic_data = pd.read_csv('titanic_train.csv')

print(titanic_data.head(20))

print(titanic_data.describe())

print(titanic_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False))

print(train.isnull().sum())

titanic_data.drop('Cabin', axis=1, inplace = True)
titanic_data.drop('Fare', axis=1, inplace = True)
titanic_data.drop('Ticket', axis=1, inplace = True)

print(titanic_data.columns)

print(titanic_data.groupby ('Survived').size())

x = titanic_data[['Survived']]
y = titanic_data['Survived']

x, x_test, y, y_test = train_test_split(x, y, test_size = 0.3)

model = LinearRegression()
print(model.fit(x, y))