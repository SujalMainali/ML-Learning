#We are going to learn the support vector machine algorithm
#we are going to classify the species of flower by considering the length  of 

#The basic idea of building a model to solve such classification problem is to build a certain boundary in graph where the one side corresponds to 
# certain species and the other side corresponds to different species

#To optimize the boundary line the distance of each line from the closest datapoint is maximized

#in case of 2D i.e two variables the boundary will be a line
#in case of 3D i.e three variables the boundary will be a plane
#For n number of features the boundary will be a hyperspace


#There are certain terms related with SVM

#while creating the decision boundary we can either consider only the nearest datapoints to build the boundary which will result in a high gamma 
#or consider even the far away datapoints which results in a low gamma
#both approaches are right the high gamma may be more computationally intensive

#Similarly the regularization refers to making the boundary more closly fit the model or make it less closely
#in case of high regularization it might result in overfitting the model to train dataset 

#In case the datapoints are mixed and proper boundary cannot be determined from the datapoints. Then, the datapoints are transformed into higher dimension using
# a mapping function. THe boundarty will then be easier to identify in the higher dimension. To avoid having to calculate the higher dimension data points the kernel
#trick is used to find the dot product of points in the higher dimension Which is sufficient for updated SVM optimization algorithm.

#we will learn more theoritical concept on how the optimization works. for now lets learn some codeing

import pandas as pd
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris= load_iris() #this load_iris library consists of large sample of  data on the iris flower

# print(dir(iris))

# print(iris.feature_names)
# print(iris.data)

df= pd.DataFrame(iris.data,columns=iris.feature_names)

df['target']=iris.target


#Here target 
#0 means setosa
#1 means versicolor
#2 means virginica

df['flower_name']=df['target'].apply(lambda x:iris.target_names[x])


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

# print(df0.head(5))
# print(df1.head(5))
# print(df2.head(5))

#we will split our dataset and try to find the feature that properly seperates the data into different flower species
#we plotted different features against each other which helps us visualize our data

#  plt.xlabel("Sapal Length")
# plt.ylabel("Sepal Width")
# plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],marker= '+', color='green')
# plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],marker= '+', color='blue')

#now lets split the dataset and train our model

feature=df.drop(['target','flower_name'],axis=1)
target= df.target

x_train,x_test,y_train,y_test= train_test_split(feature,target,test_size=0.2)

model=SVC()

print(model.fit(x_train,y_train))


print(model.score(x_test,y_test))














