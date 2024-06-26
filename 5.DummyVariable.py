#We are going to build a model to predict home prises based on area and name of town
#we can learn to handle tetx based data with this exercise

#A basic assumption what we can make is to replace the town name for each with certainn integer value but this can result in unnecessary assumptions by computer


#Today we are dealing with categorical variables i.e some text is used to divide the data into crertain category in this case the name of the town
#This type of  variable is nominal because it inherintly cant have a numeric value


#We use on hot encoding approach to deal with such data
#We create a new column for each of the categories and assign 1 or 0 based on the presense or absense of that category
import pandas as pd

df=pd.read_csv('Programming/Learning_python/Machine_Learning/Encodinghomeprices.csv')
#we have a function in pandas that can create dummy variables for each category
pd. get_dummies(df.town)
df=pd.concat([df,pd. get_dummies(df.town)])

df.drop(['town','west windsor'],axis=1)#we have to drop the initial categorya nd one of the dummy variables

from sklearn.linear_model import LinearRegression
model=LinearRegression()
#To simplify the learning method we can pass dataframes to the fit method instead of specifying the array
x=df.drop(['price'],axis='columns')#Now the x dataframe only has area and the townships
y=df.price#The y dataframe only has price

model.fit(x,y)

model.predict(['3400',0,1])#3400 area home in robinsville

#we can easily check the accuracy of the model by using score()
model.score(x,y)#this finds the predicted values for all x and compares it with real value in y