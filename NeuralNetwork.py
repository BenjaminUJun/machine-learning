import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

# Creates a 3-layer neural network with backpropagation for learning
# Emulates a certain function defined by user

def function(x,y):
	return np.tanh((x+y)/(x-y))

# Randomly generate x's and y's

n = 5000

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
print X
print y

# Parse into training and test data sets
#np.random.seed(1)
r = 0.9
index_train = np.random.choice([True,False],n,p=[r,1-r])
index_test = ~ index_train

X_train = X[index_train]
X_test = X[index_test]
y_train = y[index_train]
y_test = y[index_test]


# Build neural network

# Set hyperparameters on a 3-layer architecture: one hidden layer
L = 10 # number of nodes in hidden layer
nu = 0.0005

# Construct weight matrices
W1 = np.random.randn(X_test.shape[1],L)
W2 = np.random.randn(L,1)

iteration = 0
errors =[]
while iteration <= 50000:
	# Feed forward
	S1 = X_train.dot(W1)
	Z1 = np.tanh(S1)
	Z1_prime = (1. - np.tanh(S1) ** 2).T

	S_out = Z1.dot(W2)
	Z_out = np.tanh(S_out)

	# Compute error
	error = 0.5 * (y_train - Z_out).T.dot(y_train - Z_out)
	iteration += 1
	print "Iteration:", iteration,"error:",error[0,0]
	errors.append(error[0,0])

	# Backpropagation

	# Compute final delta
	delta_out = (Z_out - y_train).T

	# Recursively compute previous deltas
	delta_1 = Z1_prime * W2.dot(delta_out)

	# Update weights
	delta_W1 = - nu * (delta_1.dot(X_train)).T
	delta_W2 = - nu * (delta_out.dot(Z1)).T

	W1 += delta_W1
	W2 += delta_W2

print "W1\n",W1
print "W2\n",W2
print "In-sample error:", error[0,0]

S1_test = X_test.dot(W1)
Z1_test = np.tanh(S1_test)
S_out_test = Z1_test.dot(W2)
Z_out_test = np.tanh(S_out_test)

error = 0.5 * (y_test - Z_out_test).T.dot(y_test - Z_out_test)
print "Out-of-sample error:", error[0,0]

errors = np.array(errors)
plt.plot(np.arange(iteration),errors)
plt.show()


