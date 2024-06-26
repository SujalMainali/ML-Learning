#hhere we are attempting to create a  SVM classifier algorithm using numpy module
import numpy  as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self,learning_rate=0.01,lambda_param=0.01,iters=1000)  :
        self.lr=learning_rate
        self.lambda_=lambda_param
        self.w=None
        self.b=None
        self.iters=1000

    def checkWeights(self):
        return self.w
    
    def checkBias(self):
        return self.b

    def fit(self,X,y):
        n_samples,n_features= X.shape

        self.w=np.zeros(n_features)
        self.b=0

        for i in range(self.iters) :
            for idx,x_i in enumerate(X):
                condition= y[idx]*(np.dot(self.w,x_i)-self.b) >=1
                if (condition):
                    wd=2*self.lambda_*self.w
                    self.w-=self.lr*wd
                else:
                    wd=2*self.lambda_*self.w - y[idx]*x_i
                    self.w-=self.lr*wd

                    bd=y[idx]
                    self.b-=self.lr*bd


    def predict(self,X=np.array([])):
        n_sample,n_features=X.shape
        y=np.zeros(n_sample)
        for idx,xi in enumerate(X):
            y[idx]= np.dot(self.w,xi)-self.b
        return np.sign(y)
    
    def accuracy(self,X,Y):
        correct_num=0
        incorrect_num=0
        for idx,xi in enumerate(X):
            yi=np.dot(self.w,xi)-self.b
            if(Y[idx]==np.sign(yi)):
                correct_num=correct_num+1
            else:
                incorrect_num+=1
        total_predictions= len(Y)
        print(f"Correct: {correct_num} \nIncorrect:{incorrect_num} \n total: {total_predictions}")
        score= correct_num/total_predictions
        return score 
    
    def plot(self,X,Y,x_axis='Feature1',y_axis='True Value',z_axis='Feature2') :
        if isinstance(X, np.ndarray):
        # Check the number of dimensions
            if (X.ndim == 1):
                self.Plot_2D(X,Y,x_axis,y_axis)
            elif (X.ndim == 2):
                # Check if the second dimension is 3 (indicating 3D coordinates
                self.Plot_3D(X,Y,x_axis,z_axis,y_axis)
            else:
                raise ValueError("Unsupported shape for data")
        else:
            raise TypeError("Data should be a numpy array")

    def Plot_2D(self,X,Y,x_axis,y_axis):
        # for idx,xi in enumerate(X):
        #     x0=X[idx][0]
        
        mask = Y == 1
        plt.scatter(X[mask], Y[mask], marker='+', color='red')
        plt.scatter(X[~mask], Y[~mask], marker='+', color='blue')
        
        plt.xlabel=x_axis
        plt.ylabel=y_axis
        plt.legend()
        plt.show()

    def Plot_3D(self,X,Y,x_axis="Feature1",y_axis='Feature2',z_axis="Value"):
        x0=X[:,0]
        x1=X[:,1]
        mask=Y==1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x0[mask],x1[mask], Y[mask], marker='+', color='red')
        ax.scatter(x0[~mask],x1[~mask], Y[~mask], marker='+', color='blue')
        ax.set_title('PLot')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        plt.show()




                    





