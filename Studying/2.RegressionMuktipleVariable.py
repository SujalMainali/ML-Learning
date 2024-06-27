#we willl learn to predict prises of homes based on more than one factor

#Lets do a problem where the data is not completely perfect
#We have to slightly analuze the data and make sure that some kind of linear relationship exists between the datapoints and the final calculated variable

#mathematically, the prediction will be made based on the following equation:
# predicted_value= m1*first_factor + m2*second_factor + m3*third_factor + const
#the factors are also called features which are certain independent variables

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("homeprices2.csv")
#this dataframe contains a NAN value lets rectify that by filling it with sutiable data

#we can use various methods lets go with median for this one
df.fillna({
    'bedrooms': df.bedrooms.median()
},inplace=True)


#now that our data is ready Lets train the model
reg=linear_model.LinearRegression()  #This creates a linear model object and an object of LinearRegression class
reg.fit(df[['area','bedrooms','age']],df.price)
#This finally creates a linear equation which best fits the given data

