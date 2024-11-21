import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_set=pd.read_csv(r"C:\Users\TEMP\Desktop\Dataset\50_Startups.csv")
x=data_set.iloc[:, :-1].values
y=data_set.iloc[:,1].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x= LabelEncoder()
x[:,2]- labelencoder_x.fit_transform(x[:,2])
onehotencoder =OneHotEncoder()
x=onehotencoder.fit_transform(x).toarray()
print(x)
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(x, y, test_size= 0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)
#Predicting the Test set result;
y_pred=regressor.predict(x_test)
print('Train Score: ', regressor.score(x_train,y_train))
print('Test Score: ', regressor.score(x_test,y_test))
fig=plt.figure(figsize=(10, 6))
plt.scatter(y_test,y_pred)
plt.xlabel('Position Levels')
plt.ylabel('Predicted values')
plt.title('Actual vs. Predicted Values')
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],linestyle='--',color='blue')
plt.show()