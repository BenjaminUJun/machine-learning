# K-Nearest Neighbors Algorithm

from sklearn.datasets import load_iris
from sklearn import cross_validation
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset and partition into training and testing data
iris = load_iris()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4)

# Create empty distance matrix: query vector i to training vector j
dist = np.zeros((X_test.shape[0],X_train.shape[0]))

# Calculate distances and fill up distance matrix
for i in range(X_test.shape[0]):
	for j in range(X_train.shape[0]):
		d = X_test[i] - X_train[j]
		dist[i,j] = d.T.dot(d) ** 0.5

# Sort nearest neighbors, user-defined k
k = 5
count = 0

for i in range(60):
	a = y_train[np.argsort(dist[i])][:k] # stores array of k-nearest neighbor labels
	print a, stats.mode(a)[0][0], y_test[i] # prints most frequent member of KNN arrays and compares versus test label
	if stats.mode(a)[0][0] == y_test[i]:
		count +=1
print count, "/", 60

