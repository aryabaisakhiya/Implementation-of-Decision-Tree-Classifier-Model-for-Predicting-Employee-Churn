# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree classification in dataset.
4.calculate Accuracy,data prediction.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Arya Baisakhiya
RegisterNumber:212222040019  
*/
import pandas as pd
data=pd.read_csv("/content/Employee (1).csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company",
          "Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![318874928-e0a7971b-4af6-4e54-9aeb-a44a03edfdcb](https://github.com/aryabaisakhiya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393645/81622bb6-fec8-446a-ae9d-141a7e46bb89)
![318875272-e5786d97-2d64-4e2f-a91b-6af899ab22f6](https://github.com/aryabaisakhiya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393645/c8b68add-7bf9-4b73-86a1-aa04ac611de1)

![318875644-4410659a-5932-475b-a683-88b62ab650f7](https://github.com/aryabaisakhiya/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119393645/90fbdfe1-18c5-4de0-ab35-e27a7d3f8cfb)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
