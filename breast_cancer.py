import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

# clean data
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# initialize features and label
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train classifier using KNN algorithm
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

# test accuracy of the classifier
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

# create example
example = [[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 4, 1, 8, 4, 10, 2, 3, 1]]
example = np.array(example)
example = example.reshape(len(example), -1)
print('Data:', example)

# make prediction using trained classifier
prediction = clf.predict(example)
print('Prediction:', prediction)
