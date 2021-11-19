import numpy as np
import pandas as pd
import pickle
import streamlit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics

df = pd.read_csv("Data/Real-Data/Real_Combine.csv")
df = df.drop_duplicates()
df = df.dropna()
cols = list(df.columns)
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1

df1 = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

X = df1.iloc[:,:-1]
y = df1.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

rand_reg = RandomForestRegressor()
rand_reg.fit(X_train,y_train)

print("The Coefficient of determination R^2 on training dataset {}".format(rand_reg.score(X_train,y_train)))
print("The Coefficient of determination R^2 on test data {}".format(rand_reg.score(X_test,y_test)))

rand_cross = cross_val_score(rand_reg,X,y,cv=5)
print("The Mean value of Cross validation Score for AQI data is {}".format(rand_cross.mean()))
rand_pred = rand_reg.predict(X_test)
print("-------------------------------------------------------------------")
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,rand_pred))
print("Mean Squared Error:",metrics.mean_squared_error(y_test,rand_pred))
print("Root Mean Squared Error:",np.sqrt(metrics.mean_squared_error(y_test,rand_pred)))
print("-------------------------------------------------------------------")

filename = 'trained_model.sav'
pickle.dump(rand_reg,open(filename,'wb'))

loaded_model = pickle.load(open('trained_model.sav','rb'))

input_data = [2.0,7.4,9.8,4.8,1017.6,93.0,0.5,4.3]
input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)