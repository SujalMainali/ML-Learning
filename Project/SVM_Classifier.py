#hhere we are attempting to create a  SVM classifier algorithm using numpy module
import numpy  as np
import matplotlib.pyplot as plt

class SVM: #this creates a class SVM which will consists of all the variables and function which will allow us to run SVM algorithm

    def __init__(self,learning_rate=0.01,lambda_param=0.01,iters=1000)  : #this is the constructor which can accept initial parameters when object is  created
        self.lr=learning_rate
        self.lambda_=lambda_param
        self.w=None
        self.b=None
        self.iters=1000

    def checkWeights(self): #a simple function to show weights if user wishes to see it
        return self.w
    
    def checkBias(self): # similar function for bias
        return self.b

    def fit(self,X,y): #this is the main function which will perform the optimation of weights and biases

        n_samples,n_features= X.shape # the shape of x which consits of samples in rows and different features in columns is recorded as such

        self.w=np.zeros(n_features)   #Initially initializing the weights as all zeros with the no of rows same as features of X
        self.b=0                       #Initializing b as 0

        for i in range(self.iters) : #runs iteratio for certain time
            for idx,x_i in enumerate(X): #the enumerate(X) returns the index and the value of x as a tuple value for each row of x
                condition= y[idx]*(np.dot(self.w,x_i)-self.b) >=1 #Basic condition of SVM algorithm
                if (condition):
                    wd=2*self.lambda_*self.w
                    self.w-=self.lr*wd
                else:
                    wd=2*self.lambda_*self.w - y[idx]*x_i
                    self.w-=self.lr*wd

                    bd=y[idx]
                    self.b-=self.lr*bd


    def predict(self,X=np.array([])): #This function is called after the user has fitted the data then this predict function is used to make predictions
        n_sample,n_features=X.shape
        y=np.zeros(n_sample)
        for idx,xi in enumerate(X):
            y[idx]= np.dot(self.w,xi)-self.b
        return np.sign(y)
    
    def accuracy(self,X,Y): #this checks the accuracy of prediction by counting the no of correct and incorrect predictions the score is ratio of correct to total predictions
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
    
    def plot(self,X,Y,x_axis='Feature1',y_axis='True Value',z_axis='Feature2') : #this is a plot function created to make life easier to plot 2D and 3D plots which working with some data

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
        #This algorithm plots x and y as red when Y==1 and blue when y==-1
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




                    





