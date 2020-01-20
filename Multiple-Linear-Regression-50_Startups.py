#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Datasets
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,4]

#converting State name column into categorical column
states = pd.get_dummies(X['State'],drop_first=True)

#Remove that state column from dataset
X=X.drop('State',axis=1)

#Add the categorical state column in the dataset
X=pd.concat([X,states],axis=1)

#splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 0)

#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x_train = sc_x.fit_transform(x_train)
#x_test = sc_x.transform(x_test)

#Our MultipleLinear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
model = LinearRegression()
model.fit(x_train,y_train)

#prediction of test set
y_pred = model.predict(x_test)

#Test_Score
from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)
