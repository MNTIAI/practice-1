import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
# Load dataset
data = pd.read_csv(r"C:\Users\TEMP\Desktop\Dataset\User_Data.csv")
X, y = data.iloc[:, [2, 3]].values, data.iloc[:, 4].values
# Split and scale the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the Decision Tree classifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)
# Predict and evaluate
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# Visualize the results
x1, x2 = np.meshgrid(np.arange(X_train[:, 0].min() - 1, X_train[:, 0].max() + 1,
0.01),np.arange(X_train[:, 1].min() - 1, X_train[:, 1].max() + 1, 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),
x2.ravel()]).T).reshape(x1.shape),alpha=0.75, cmap=ListedColormap(('purple', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(X_train[y_train == j, 0], X_train[y_train == j, 1],c=ListedColormap(('purple', 'green'))(i), label=j)
    plt.title('Decision Tree Algorithm (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()