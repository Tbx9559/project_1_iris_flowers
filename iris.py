# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 18:24:57 2023

@author: Thibaut R
"""
# import all librairies, components used for analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# opening and reading the file
ds=pd.read_csv("C:/Users/phili/IRIS_ Flower_Dataset.csv")
#quickly show the first 5 ines
ds.head(5)
#reading information about the file data
ds.info()
#Verification of the existence of null values to be able to eliminate them.
ds.isnull().sum()
#Here there are no errors


#Distribution of flower types, here fair
ds.columns
ds['species'].value_counts()
sb.countplot(ds['species'])
#Choice of parameters of x, here all data execept the name of flowers
x=ds.iloc[:,:4]
#Choice of y parameters, here only the names = the answer to give
y=ds.iloc[:,4]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
#Creation of the random forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)
#Using the fit function to imrove the accuracy of the random forest
random_forest.fit(x_train,y_train)
y_pred=random_forest.predict(x_test)
res=random_forest.score(x_train,y_train)
#Result
print(res*100,2)

#Interpret results based on file data

#Fix random forest for more realistic results
from sklearn.model_selection import cross_val_score
rf=RandomForestClassifier(n_estimators=100)
scores=cross_val_score(rf,x_train,y_train,cv=10,scoring="accuracy")
#Result
plt.ylabel("Scores")
plt.title("Random Forest test")
plt.plot(scores)
plt.plot(res)
plt.show()
#Accuracy/average
print(scores)
print(scores.mean())
print(scores.std())

#Distribution of flower data to show deviations
data1=ds['sepal_length']
data2=ds['sepal_width']
data3=ds['petal_length']
data4=ds['petal_length']
data=[data1,data2,data3,data4]
plt.boxplot(data)
plt.title("Iris data discrepancy")
plt.show()

#
plt.figure(figsize=(8, 6))
sb.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
from joblib import dump, load
dump('titanic_rdf_model_save.joblib')

