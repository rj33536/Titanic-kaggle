from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

def get_err(train_X,train_y,val_X,val_y,leaf):
    model = RandomForestRegressor(max_leaf_nodes=leaf)
    model.fit(train_X,train_y)
    predicted =model.predict(val_X)
    return (mean_absolute_error(predicted,val_y))

train=pd.read_csv("train.csv")
print(train.columns)
print(train.describe())
prediction_col=['Pclass','Sex' ,'SibSp','Parch','Fare','Embarked']
X=train[prediction_col]
X = pd.get_dummies(X)
print(X.head())
y=train.Survived
my_imputer=Imputer()
X=my_imputer.fit_transform(X)
train_X,val_X,train_y,val_y=train_test_split(X,y,random_state=0)
leaf=100
min_err=1
for i in range(50,500):
    err=get_err(train_X,train_y,val_X,val_y,i)
    if err<min_err:
        min_err=err
        leaf=i

print(err)
print(i)
model = LogisticRegression()
model.fit(train_X,y)
test = pd.read_csv('test.csv')
test_X=test[prediction_col]
test_X=pd.get_dummies(test_X)
test_X=my_imputer.fit_transform(test_X)
predicted=model.predict(test_X)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted})
my_submission.to_csv('submission.csv', index=False)


