from Project.GradientDescent import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt


Model=gradient_descent()
df=pd.read_csv("Studying/homeprices2.csv")
df.dropna(inplace=True)
print(df)

