# Adaptive gradient algorithm Perceptron Learning Algorithm
import numpy as np
import matplotlib.pyplot as plt

# Define model
def line(t,B0,B1,B2):
	return -(B0 + B1 * t)/B2
n = 1000

# Generate linearly separable data
z_plus = np.ones((n,1))
z_minus = np.ones((n,1))

a = 20 * np.random.randn(n,2) + 50# label 1
b = 20 * np.random.randn(n,2) - 50 # label -1

z_plus = np.concatenate((z_plus,a),axis=1)
z_minus = np.concatenate((z_minus,b),axis=1)

# Classify data
z_plus = np.concatenate((z_plus,np.ones((n,1))),axis=1)
z_minus = np.concatenate((z_minus,np.zeros((n,1))),axis=1)
z_full = np.concatenate((z_minus,z_plus),axis=0)

# Separate into training and testing sets, training sample size r ratio
r = 1
index_train = np.random.choice([True,False],2*n,p=[r,1-r])
index_test = ~ index_train

z_train = z_full[index_train]
z_test = z_full[index_test]

# Set range for separating plane
x_min_range = min(min(a[:,0]),min(b[:,0]))
x_max_range = max(max(a[:,0]),max(b[:,0]))
t = np.arange(x_min_range,x_max_range)

# Initialize perceptron parameters with preconditioning
alpha = 0.1
B0 = 0.
B1 = 0.
B2 = 0.
threshold = 0.5
w = np.array([B0,B1,B2])
y = z_train[:,3]
z_train = np.delete(z_train,3,1)
N = len(z_train)

k = 0
error_count = 1
while error_count != 0:
	print '-' * 60
	error_count = 0
	delta = np.zeros(N)
	for i in range(N):
		k += 1
		result = z_train[i].dot(w) > threshold
		error = y[i] - result
		if error != 0:
			error_count += 1
			d = alpha * error
			for j in range(len(w)):
				w[j] += d*z_train[i,j]	
	print k, w


B0 = w[0]
B1 = w[1]
B2 = w[2]
plt.plot(a[:,0],a[:,1],'ro',b[:,0],b[:,1],'bo')
plt.plot(t,line(t,B0,B1,B2))
plt.show()