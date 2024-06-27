#we are going to solve classification problem using decision tree

#In this example we will try to predict if  a person salary is more than 100k based on three features
#we will also learn label encoder which we ignored previously

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df=pd.read_csv("9.salaries.csv")
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']

#create three labeencoder object for three columns
label_company= LabelEncoder()
label_job= LabelEncoder()
label_degree= LabelEncoder()

#This creates the required new columns and encodes them with necessary numeric values to get rid of the named category
inputs['CompanyCoded']= label_company.fit_transform(inputs['company'])
inputs['JobCoded']= label_company.fit_transform(inputs['job'])
inputs['DegreeCoded']= label_company.fit_transform(inputs['degree'])

Final_Input= inputs.drop(['company','job','degree'],axis=1)

#here in final_input google is assigned as 2, abc 0, and facebook 1
#sales executive 2, business manager 0 , programmer 1
# bachelor 0 and master 1

model= DecisionTreeClassifier() #This creates a decision tree classifier object

x_train,x_test,y_train,y_test=train_test_split(Final_Input, target,test_size=0.2,random_state=6)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)

print(y_predict)

cm=confusion_matrix(y_test,y_predict)

