import pandas as pd
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

digits=load_digits()

#print(dir(digits))

# print(digits.data)

df=pd.DataFrame(digits.data,columns= digits.feature_names)


df['target']=digits.target

df['target_names']=df.target.apply(lambda x: digits.target_names[x%10])

#print(digits.target[0:20])
#print(digits.target_names)
#print(df)
def give_proper_target():
    pass

#now that our dataframe is ready lets work on the model

Model=SVC()

features=df.drop(['target','target_names'],axis=1)
target=df['target']

x_train,x_test,y_train,y_test= train_test_split(features,target,test_size=0.2)

Model.fit(x_train,y_train)

print(Model.score(x_test,y_test))
