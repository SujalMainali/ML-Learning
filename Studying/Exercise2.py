import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

df= pd.read_csv("Excise2.csv")

df['experience'] = df['experience'].apply(w2n.word_to_num)
#This will convert the numbers in words to numerical forms

#we can now focus on filling missing data

#For this example i will put mean of the rest of the data on missing data
df.fillna({
    'experience':df['experience'].mean(),
    'test_score(out of 10)': df['test_score(out of 10)'].mean()
})

#Now that the data is ready we can now create the model
reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($s)'])