# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# Loading data
data = pd.read_csv("iris.csv")


# Analysing and Visualising Data
for col in data:
    print(col)
plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
sns.boxplot(x='class', y='sepallength', data=data)
plt.subplot(2, 2, 2)
sns.boxplot(x='class', y='sepalwidth', data=data)
plt.subplot(2, 2, 3)
sns.boxplot(x='class', y='petallength', data=data)
plt.subplot(2, 2, 4)
sns.boxplot(x='class', y='petalwidth', data=data)
plt.show()


# Splitting up dataset
array = data.values
x = array[:, 0: 4]
y = array[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state=0)


# Applying Support Vector Classifier and evaluation
model = SVC(max_iter=1000, gamma='auto')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = round(accuracy_score(y_pred, y_test), 2) * 100
print(accuracy)