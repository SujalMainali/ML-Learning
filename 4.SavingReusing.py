#Today we will learn to save a trained model to use later

#The pickel module allows us to serialize a python object to a file
import pickle
from sklearn import linear_model
import pandas as pd


df=pd.read_csv("Programming/Learning_python/Machine_Learning/homeprices.csv")
reg=linear_model.LinearRegression()#THis will create a regression linear model
reg.fit(df[['area']],df.price)
reg.predict([[2700]])

with open('HousePrises_model','wb') as f:
    pickle.dump(reg,f)

with open('HousePrises_model','wb') as f:
    model=pickle.load(f)

