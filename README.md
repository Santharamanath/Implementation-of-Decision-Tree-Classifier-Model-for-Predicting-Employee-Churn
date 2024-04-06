# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Santha ramanath M

RegisterNumber: 212223220097 
*/
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,206,6,0,1,2]])
```

## Output:

# Data Head:

![318633191-13726f73-a3be-462d-a126-276b0d8fee29](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/d7c7ba51-b0c1-4533-8148-15babd5025cf)

# Dataset info:

`![318633273-1b111b8d-93df-481a-a44a-39148c64d82a](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/e5eb7eb9-4eed-4f66-abbd-873bf0a98893)

# Null Dataset:

![318633380-31fbbe21-fa4e-4614-a24d-a126ec4ae050](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/bd45fc1e-dbcd-4df4-a4eb-134424faa703)

# Values count in left column:

![318633530-cf6320ac-df7f-48d7-bfc5-bbee674598f1](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/e8bfe98c-2d56-41bc-a3ba-4c9bae08a395)

# Dataset transformed head:

![318633660-5bcbbc7b-4e7f-41dd-82c0-3e59250ea9cd](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/5b76c3c8-c833-423f-b826-bed7db1bb5c2)

# x.head:

![318633777-f9e0a1bf-e3d8-4c08-bba0-b9192176d3b1](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/61f2c3ee-6220-4efa-853f-0af6b8332324)

# Accuracy:

0.985

# Data Prediction:

![318634020-0be83ccc-14db-484a-a57c-6af6cef509e7](https://github.com/Santharamanath/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149035289/441719a0-d664-4283-ae45-c03d741ccd75)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
