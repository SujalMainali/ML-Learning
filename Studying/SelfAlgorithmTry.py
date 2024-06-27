from Project.SVM_Classifier import SVM

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df=pd.read_csv("/home/violent_sujal67/Programming/Learning_python/Machine_Learning/Project/SVM.csv")
#print(df)

model=SVM()
features=df[['Sex','Age']]
target= df['Survived']

x_train,x_test,y_train,y_test= train_test_split(features,target,random_state=40)

x_train_np= np.array(x_train)
x_test_np= np.array(x_test)
y_train_np= np.array(y_train)
y_test_np= np.array(y_test)

# model.fit(x_test_np,y_train_np)

# print(model.predict(np.array([[1,22]])))

model.plot(x_train_np,y_train_np)
model.plot(x_test_np,y_test_np)


