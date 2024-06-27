#We can split the dataset into a test and train dataset using sklearn train_test_split
from sklearn.model_selection import train_test_split
#x and y are the respective dependent features and independent variables
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)#This retrns four dataset which are stored in the four variables

#The train_test_split() sp;its the data randomly to ensure maximum accuracy of the model

#we can specify random_test=10 The dataset remains same
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)




