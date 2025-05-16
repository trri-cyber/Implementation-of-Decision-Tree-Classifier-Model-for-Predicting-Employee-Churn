# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas 2. Import Decision tree classifier 3. Fit the data in the model 4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Rishab p doshi
RegisterNumber:  212224240134
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('/content/Employee.csv')
print(df.head())

print(df.info())
print(df.isnull().sum())
print("\n",df["left"].value_counts())
le = LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])

print(df.head())
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

y=df["left"]
print(x.head())
print(y.head())
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(f"accuracy: {accuracy_score(y_pred,y_test)}")
plt.figure(figsize=(15,10))
plot_tree(model,filled=True,feature_names=x.columns,class_names=['stayed','Left'])
plt.show()
```
## Output:
![Screenshot 2025-05-16 130226](https://github.com/user-attachments/assets/458c882f-50dd-4969-abdd-d861fc967fd5)
![Screenshot 2025-05-16 130250](https://github.com/user-attachments/assets/b1a7b78e-8aca-479e-ae27-27957a953f61)
![image](https://github.com/user-attachments/assets/22553757-1636-4ed6-969a-dd3ca02a2bc7)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
