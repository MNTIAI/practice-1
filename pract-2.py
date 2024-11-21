import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data_set=pd.read_csv(r"C:\Users\TEMP\Desktop\Dataset\Salary_Data.csv")
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,1].values
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
x_pred=regressor.predict(x_train)
x_pred
plt.scatter(x_train,y_train, color="green")
plt.plot(x_train,x_pred,color="red")
plt.title("Salary vs Experience (TrainingDataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(In Rupees)")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,x_pred,color="red")
plt.title("Salary vs Experience (TestDataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary(In Rupees)")
plt.show()