#here we are going to predict the survival of a passanger in titanic

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('exercise9.csv')
df.drop(['Name','SibSp','Parch','Ticket','Embarked','Cabin','PassengerId','Pclass'],axis=1,inplace=True)

df.dropna(inplace=True)

label_sex= LabelEncoder()
Sex_label= label_sex.fit_transform(df['Sex'])
df['Sex_Encoded']= Sex_label
df.drop(['Sex'],axis=1,inplace=True)

input=df.drop(['Survived'],axis=1)
target=df['Survived']

x_train,x_test,y_train,y_test= train_test_split(input,target,test_size=0.2)


Model=DecisionTreeClassifier()
Model.fit(x_train,y_train)

Model.score(x_test,y_test)