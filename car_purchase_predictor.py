import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

dataset = pd.read_csv("C:/Users/hacho/Downloads/Machine-Learning/Logistic-Regression-Classification/Social_Network_Ads.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print('Car Purchase Predictor')
print('----------------------')
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')

def predict_purchase(age, salary):
    transformed_data = sc.fit_transform([[age, salary]])
    result = classifier.predict(transformed_data)
    if result == [0]:
         print(f'This {age} year old customer with a salary of ${salary} will likely NOT purchase the car')
    elif result == [1]:
        print(f'This {age} year old customer with a salary of ${salary} WILL likely purchase the car')
    
age = float(input('Enter Age '))
salary = float(input('Enter Salary '))
predict_purchase(age, salary)