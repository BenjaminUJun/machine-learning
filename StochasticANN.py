import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Creates a k-layer artificial neural network using backpropagation
# Accepts a scalar input k defining number of hidden layers
# Minimization of error is through stochastic gradient descent
# Accepts a k-dimensional vector defining the number of nodes in the ith indexed component
# corresponding to the number of nodes in that layer

# This version is more general since it includes bias nodes to each hidden layer
# Tends to get stuck in local minima but generally can be trained

def function(x,y):
	return np.tanh(x*y)

# Randomly generate x's and y's

n = 1000

x = np.random.randn(n,1)
y = np.random.randn(n,1)

X = np.concatenate((x,y),axis=1)

z = function(x,y)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(x,y,z)
#plt.show()

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
r = 0.9
index_train = np.random.choice([True,False],n,p=[r,1-r])
index_test = ~ index_train

X_train = X[index_train]
X_test = X[index_test]
y_train = y[index_train]
y_test = y[index_test]

# Create minibatches; determine batch size

index_batch = np.random.choice([True,False],X_train.shape[0],p=[0.2,0.8])
print len(index_batch)
X_batch =  X_train[index_batch]
y_batch = y_train[index_batch]

# Build neural network

# Set hyperparameters, k as the number of hidden layers
k = 3
nodes = [8,4,2]
print "k = ",k
print "nodes =", nodes

if k != len(nodes):
	print "Error: nodes length not equal to number of hidden layers"
nu = 0.0001
print "nu = ", nu

# Construct weight matrices 
W = [np.random.randn(X_batch.shape[1],nodes[0])]
for i in range(1,k):
	W.append(np.random.randn(nodes[i-1]+1,nodes[i]))
W.append(np.random.randn(nodes[k-1]+1,y_batch.shape[1]))

for i in range(len(W)):
	print "W[%i]" %(i), W[i].shape

iteration = 0
errors = []
errors_validation = []
while iteration <= 100000:
	# Create minibatches; determine batch size
	index_batch = np.random.choice([True,False],X_train.shape[0],p=[0.2,0.8])
	X_batch =  X_train[index_batch]
	y_batch = y_train[index_batch]

	# Feedforward
	temp = np.tanh(X_batch.dot(W[0]))
	temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
	Z = [temp]
	Z_prime = [(1. - (np.tanh(X_batch.dot(W[0]))) ** 2).T]

	for i in range(1,k+1):
		temp = np.tanh(Z[i-1].dot(W[i]))
		temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
		Z.append(temp)
		temp = (1.-(np.tanh(Z[i-1].dot(W[i]))) ** 2).T
		Z_prime.append(temp)

	y_hat = np.tanh(Z[k-1].dot(W[k]))

	# Compute errors
	error = 0.5 * (y_batch - y_hat).T.dot(y_batch - y_hat)
	iteration += 1
	print "Iteration:", iteration,"error:",error[0,0]
	errors.append(error[0,0])

	# Backpropagation
	delta_hat = (y_hat - y_batch).T
	delta = [Z_prime[k] * W[k][1:,:].dot(delta_hat)]
	for i in range(1,k):
		delta.append(Z_prime[k-i-1] * W[k-i][1:,:].dot(delta[i-1]))

	# Update weights
	deltaW = [-nu * (delta[k-1].dot(X_batch)).T]
	for i in range(1,k):
		deltaW.append(-nu * (delta[k-i-1].dot(Z[i-1])).T)
	deltaW.append(-nu * (delta_hat.dot(Z[k-1])).T)

	for i in range(k):
		W[i] += deltaW[i]

	# Track validation error
	temp = np.tanh(X_test.dot(W[0]))
	temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
	y_test_hat = temp

	for i in range(1,k):
		temp = np.tanh(y_test_hat.dot(W[i]))
		temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
		y_test_hat = temp
	y_test_hat = np.tanh(y_test_hat.dot(W[k]))
	error_test = 0.5 * (y_test_hat - y_test).T.dot(y_test_hat - y_test)
	errors_validation.append(error_test[0,0])


for i in range(len(W)):
	print "W[%i]\n" %(i),W[i]
print "Neural network architecture: ", nodes
print "In-sample error:", error[0,0]

temp = np.tanh(X_test.dot(W[0]))
temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
y_test_hat = temp

for i in range(1,k):
	temp = np.tanh(y_test_hat.dot(W[i]))
	temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
	y_test_hat = temp
y_test_hat = np.tanh(y_test_hat.dot(W[k]))

error_test = 0.5 * (y_test_hat - y_test).T.dot(y_test_hat - y_test)
print "Out-of-sample error:", error_test[0,0]

errors = np.array(errors)
plt.plot(np.arange(iteration),errors)
plt.plot(np.arange(iteration),errors_validation)
plt.yscale('log')
plt.show()

