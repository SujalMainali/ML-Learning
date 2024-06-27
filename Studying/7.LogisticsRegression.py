#Upto now we have dealth with ML model that deals with fixed numerical predicted value
#We may also need to predict a certain categorical output i.e classification problem for this we use logistics regression

#If there are  only two categories to classify then it is binary regression 
#This is a simple exercise that answers whether a person will or will not buy insurance based on the information provided in this case age

#Here if we draw a scatter plot between the 1 or 0 of the caregory and age then the scatter plot cant be properly fitted by a straight line so we have to move beyond linear regression.
#The scatter plot looks like sigmoid or logit function z=1/(1+e**-z)

#The proper sigmoid function can be created by simply using linear regressionmn to find the m and b. 
#Then the sigmoid function will be: y=1/(1+e**-(m*x+b))

from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df= pd.read_csv("Logistics_regression_insurance_data.csv")

plt.scatter(df.age,df.bought_insurance,marker='+')

x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance)
Reg=linear_model.LinearRegression()#This is for linear regression

Reg_logistics=linear_model.LogisticRegression()#This is for logistics regression
Reg_logistics.fit(x_train,y_train)

plt.plot(df.age,Reg_logistics.predict([df.age]))
plt.show()


