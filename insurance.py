import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics

data_set = pd.read_csv('insurance.csv')

print(data_set.head(20))

print(data_set.isnull().sum())

print(data_set.groupby ('smoker').size())

data_set['bmi'].hist(bins= 30, figsize=(10, 5))

x = data_set[["age", 'bmi', 'children']]
y = data_set["charges"]

x_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

model = LinearRegression()

model.fit(x_train, y_train)

print(model.score(x_train,y_train))