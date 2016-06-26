import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creates a k-layer artificial neural network using backpropagation
# Accepts a scalar input k defining number of hidden layers
# Minimization of error is through batch gradient descent
# Accepts a k-dimensional vector defining the number of nodes in the ith indexed component
# corresponding to the number of nodes in that layer

def function(x,y):
	return np.tanh(x*y)+np.random.randn(1)

# Randomly generate x's and y's

n = 1000

x = np.random.randn(n,1)
y = np.random.randn(n,1)

X = np.concatenate((x,y),axis=1)

z = function(x,y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x,y,z)
plt.show()

# Parse data into feature matrix and target vector

y = z

# Normalize data
for i in range(X.shape[1]):
	X[:,i] /= max(X[:,i])

y[:,0] /= max(y[:,0])

# Append bias values to X
X = np.concatenate((np.ones((n,1)),X),axis = 1)

# Parse into training and test data sets
#np.random.seed(1)
r = 0.8
index_train = np.random.choice([True,False],n,p=[r,1-r])
index_test = ~ index_train

X_train = X[index_train]
X_test = X[index_test]
y_train = y[index_train]
y_test = y[index_test]

# Build neural network

# Set hyperparameters, k as the number of hidden layers, 
k = 2
nodes = [5,3]
if k != len(nodes):
	print "Error: nodes length not equal to number of hidden layers"
nu = 0.0001

# Construct weight matrices 
W = [np.random.randn(X_test.shape[1],nodes[0])]
for i in range(1,k):
	W.append(np.random.randn(nodes[i-1],nodes[i]))
W.append(np.random.randn(nodes[k-1],y_train.shape[1]))

iteration = 0
errors = []
errors_validation = []

while iteration <= 100000:
	# Feedforward
	Z = [np.tanh(X_train.dot(W[0]))]
	Z_prime = [(1. - (np.tanh(X_train.dot(W[0]))) ** 2).T]
	for i in range(1,k):
		Z.append(np.tanh(Z[i-1].dot(W[i])))
		Z_prime.append((1.-(np.tanh(Z[i-1].dot(W[i]))) ** 2).T)

	y_hat = np.tanh(Z[k-1].dot(W[k]))

	# Compute errors
	error = 0.5 * (y_train - y_hat).T.dot(y_train - y_hat)
	iteration += 1
	print "Iteration:", iteration,"error:",error[0,0]
	errors.append(error[0,0])

	# Backpropagation
	delta_hat = (y_hat - y_train).T
	delta = [Z_prime[k-1] * W[k].dot(delta_hat)]
	for i in range(1,k):
		delta.append(Z_prime[k-i-1] * W[k-i].dot(delta[i-1]))

	# Update weights
	deltaW = [-nu * (delta[k-1].dot(X_train)).T]
	for i in range(1,k):
		deltaW.append(-nu * (delta[k-i-1].dot(Z[i-1])).T)
	for i in range(k):
		W[i] += deltaW[i]

	# Track validation error
	y_test_hat = np.tanh(X_test.dot(W[0]))
	for i in range(1,k+1):
		y_test_hat = np.tanh(y_test_hat.dot(W[i]))
	error_test = 0.5 * (y_test_hat - y_test).T.dot(y_test_hat - y_test)
	errors_validation.append(error_test[0,0])


print "W\n", W
print "In-sample error:", error[0,0]

y_test_hat = np.tanh(X_test.dot(W[0]))
for i in range(1,k+1):
	y_test_hat = np.tanh(y_test_hat.dot(W[i]))

error_test = 0.5 * (y_test_hat - y_test).T.dot(y_test_hat - y_test)
print "Out-of-sample error:", error_test[0,0]

errors = np.array(errors)
plt.plot(np.arange(iteration),errors)
plt.plot(np.arange(iteration),errors_validation)
plt.yscale('log')
plt.show()

