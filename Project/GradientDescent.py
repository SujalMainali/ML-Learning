#Here we are writing a gradient descent algorithm using numpy for the purposes of optimizing the value of weights and bias
import pandas as pd

import numpy as np

class gradient_descent: #creating a gradient descent class to store the variables and functions relating to the algorithm
    def __init__(self,learning_rate=0.01,iteration=1000): #using constructor to initialize and accept some basic values
        self.lr=learning_rate
        self.iter=iteration

    def fit(self,x,y): #fit function performs the main algorithm of gradient descent
        X=np.array(x) 
        Y=np.array(y).reshape(-1,1) #this reshapes the Y array from (features, ) to (features, 1)
        n_samples,n_features=X.shape
        self.w=np.zeros((n_features,1)) #A matrix of single column is created to store weights
        self.b= 0
        # print(f"{(self.w).shape} {self.w} \n ({n_samples},{n_features})")

        #This is to scale the data in order to make the cost function manageable
        for col_idx in range(n_features):
            if (np.max(X[:, col_idx]) > 100)and(np.max(X[:, col_idx]) < 1000):
                X[:, col_idx] /= 1000
            elif(np.max(X[:, col_idx]) > 1000)and(np.max(X[:, col_idx]) < 10000):
                X[:, col_idx] /= 10000
            elif(np.max(X[:, col_idx]) > 10000)and(np.max(X[:, col_idx]) < 100000):
                X[:, col_idx] /= 100000
        md=np.zeros((n_features,1))
        bd=0
        #print(f"{X.shape},{Y.shape},{self.w.shape}, {md.shape}")

        #This is implementing the algorithm for certain no of ierations
        for i in range(self.iter):
            
            y_predicted=np.dot(X,self.w) + self.b
            z=Y-y_predicted
            cost=(1/n_samples)* np.sum(z**2)
            
            # print(f"{md.shape}")
            md=-(2 / n_samples) * np.dot(X.T, z)
            bd=-(2/n_samples)* np.sum(z)

            
            
            self.w= self.w- self.lr*md
            self.b= self.b - self.lr*bd

            #this prints the cost and derivatives every 100 iterations
            if (i % 100 == 0):
                print(f"cost= {cost}")
                print(f"md= {md}  bd={bd}")
                print(f"Weight= {self.w} Bias={self.b}")


    def predict(self,x=pd.DataFrame()):#The predict function is used to make prediction after fitting the data
        X=np.array(x)
        n_samples,n_features=X.shape
        for col_idx in range(n_features):
            if (np.max(X[:, col_idx]) > 100)and(np.max(X[:, col_idx]) < 1000):
                X[:, col_idx] /= 1000
            elif(np.max(X[:, col_idx]) > 1000)and(np.max(X[:, col_idx]) < 10000):
                X[:, col_idx] /= 10000
            elif(np.max(X[:, col_idx]) > 10000)and(np.max(X[:, col_idx]) < 100000):
                X[:, col_idx] /= 100000
        y_pred= np.dot(X,self.w)+self.b
        return y_pred
    

df=pd.read_csv("homeprices2.csv")
df.dropna(inplace=True)
print(df)


Model=gradient_descent(learning_rate=0.0001,iteration=100000)

Model.fit(df[['area','bedrooms','age']],df.price)

print(Model.predict([[3400.0,5.0,19.0]]))



    


