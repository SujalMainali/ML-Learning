import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv("canada_per_capita_income.csv")
df.rename(columns={'per capita income (US$)' :'Income'}, inplace=True)


plt.xlabel("Year")
plt.ylabel("Per Capita Income($US)")
plt.scatter(df.year,df.Income)
plt.show()

reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.Income)