import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 


class LogisticsRegression:
    def __init__(self,learning_rate=0.01,iterations=1000):
        self.lr=learning_rate
        self.iter=iterations

    def sigmoid(self,z):
        return 1/(1+ np.exp(-z))

    def fit(self,X,Y):
        X=np.array(X)
        Y=np.array(Y).reshape(-1,1)
        n_samples,n_features=X.shape

        self.w=np.zeros((n_features,1))
        self.b=0

        for i in range(self.iter):
            z=np.dot(X,self.w)+self.b
            y_pred= self.sigmoid(z)


            cost=(-1/n_samples)* (np.dot(Y.T,y_pred)+np.dot((1-Y).T,(1-y_pred)))
            wd=(1/n_samples)*np.dot(X.T,(y_pred-Y))
            bd=(1/n_samples)*np.sum(y_pred-Y)

            self.w-=self.lr*wd
            self.b-=self.lr*bd

    def predict(self,X):
        X=np.array(X)
        n_samples,n_features=X.shape
        y_predict= self.sigmoid(np.dot(X,self.w)+self.b)
        y_pred=np.zeros((n_samples,1))
        for idx,yi in enumerate(y_predict):
            if (float(yi)>=0.5):
                y_pred[idx][0]=1
            else:
                y_pred[idx][0]=0

        return y_pred

        
    def score(self,X,Y):
        X=np.array(X)
        Y=np.array(Y).reshape(-1,1)
        y_predict= self.predict(X)
        
        Diff=Y-y_predict
        #print(f"Prediction: {y_predict}\n True values : {Y}\n Difference: {Diff}")
        correct_num=len(Diff[Diff==0])
        incorrect_num=len(Diff[Diff!=0])
        #print(f"Correct= {correct_num} Incorrect= {incorrect_num}")
        score= correct_num/(correct_num+incorrect_num)
        print(score)

df= pd.read_csv("/Logistics_regression_insurance_data.csv")
# print(df)

# Model=LogisticsRegression()
# Model.fit(df[['age']],df['bought_insurance'])
# if (Model.predict([[29]])==1):
#     print("Buys Insurance")

# else:
#     print("Doesnt buy insurance")

# x_train,x_test,y_train,y_test=train_test_split(df[['age']],df['bought_insurance'])
# Model.fit(x_train,y_train)
# Model.score(x_test,y_test)





