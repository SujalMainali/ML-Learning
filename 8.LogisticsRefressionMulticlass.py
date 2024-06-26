import matplotlib.pyplot as plt
from sklearn.datasets import load_digits#The sklearn datasets have some pre loaded datasets that we can use to learn ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

digits =load_digits()
dir(digits)

digits.data[0] #this shows the data of first element
plt.matshow(digits.images[0])

x_train, x_test, y_train, y_test=train_test_split(digits.data, digits.target,test_size=0.2,random_state=2)

#we will use logistics regression to classify the input into more than one category
#in this case each digit data can be classified as any of the possible digits
reg= LogisticRegression()
reg.fit(x_train,y_train)

reg.score(x_test,y_test) #this is used to check the accuracy of the model

y_predicted= reg.predict(x_test)

cm=confusion_matrix(y_test,y_predicted)#This shows the matrix of the truth values and predicted values. The predicted values are in x-axis and true values are in y-axis

