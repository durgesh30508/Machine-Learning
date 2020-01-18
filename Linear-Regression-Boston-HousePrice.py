#importing libraries
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

#importing data from sklearn
from sklearn.datasets import load_boston
boston = load_boston()

#Dataframes
df_x = pd.DataFrame(boston.data,columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

#DataSplitting
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=4)

#Model
reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)

#resulted weight (The term theta for each feauture)
reg.coef_
#below is the array of thetas

#prediction
a=reg.predict(x_test)

#mean square error
np.mean((a-y_test)**2)
