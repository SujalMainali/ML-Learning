#for this exercise we are building a maching learning algorithm that best predicts the prise of a home 
#We will supply it with data list consisting of the prises of home and  their size in square feet

#To solve this type of problem we have to find a straight line which most closely alligns with the data


#To find the beat line we calculate the difference or error between the value of y(price) for each given value of x(area)
#THe error is then squared and sum the line with minimum error is  correct

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("Programming/Learning python/Machine Learning/homeprices.csv")

plt.scatter(df.area,df.price)
#This will plot the points having area on x-axis and prices on y-axis

#we can also easily modify our plot to look better
plt.xlabel('area(sqr ft)')
plt.ylabel('prices($us)')
#This gives context to the data in x and y axis
plt.scatter(df.area,df.price,color= 'red',marker='+')


#The plot of the data has been created. Now we use a linear regression model
#wwe have imported linear_model from sklearn
reg=linear_model.LinearRegression()#THis will create a regression linear model
reg.fit(df[['area']],df.price)#This trains the linear regression model on available data
#we supply the data frame with individual columns

#REMEMBER that first argument must be a two dimensional array

#after tthe fit code runs sucessfully we can then predict the prices for various values of areas using predict
reg.predict([[2700]])#Here also the argument should be 2D array

reg.predict([[2700,3000,3500]])# We cannot do this because this creates a 2D array with one row and three columns and each column is treated as single sample
reg.predict([[2700],[3000],[3500]]) #this is however possible as this creates a 2D array with three rows and one column


#we can also supply a dataframe having the list of values to be predicted as it acts as 2D array

#to see the final predicted linear equation
plt.plot(df.area, reg.predict(df[['area']]))
plt.show()