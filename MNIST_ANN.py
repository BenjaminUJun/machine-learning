#
#		Recognizing Handwritten Digits with Neural Networks
#
#		MNIST_ANN.py
#
# Implementation of artificial neural network to recognize handwritten digits
# Uses MNIST training dataset of 60,000 28x28 pixel digits and 10,000 testing samples
# Data can be retrieved from http://yann.lecun.com/exdb/mnist/ 
# Converted to .csv using mnistconverter.py (see attached file)
# 
# Features v2:
# - Divided into three phases: Initialization, Training, and Testing
# - Initialization opens MNIST training data and preprocesses for training
# - Training involves the core neural network algorithm
# - Testing opens MNIST test data, applies trained model, and evaluates performance
# - Added feature to save model parameters if performance exceeds 96% on test data
# - Changed activation function to 1.7519 tanh(2/3z) in line with LeCun's Efficient Backprop guidelines
# - Added dropout procedure
# 
# Features v3:
# - Steamlined general flow into preprocessing, initialization, and testing
# - Defined recurring functions for conciseness
# - Implemented various stochastic gradient descent optimization methods, vastly improving convergence rate
# - Added progress bar when training per epoch
# - Fixed last activation layer to use softmax
# - Added 'warm start' feature to use pretrained weights from saved sessions
# - Added feature to determine batch size and cross validation split size
# 
# 
# Current performance: 98.35% (784-800-800-10 architecture)
# Goal: 99%

# Created by Miguel Benavides on September 15, 2016
#
# GitHub: https://github.com/MiguelBenavides/machine-learning
# Email: migibenavides@gmail.com

import csv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import time, sys
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

### Preprocessing

# Open MNIST Training Set
file = '/Users/miguelbenavides/Documents/PythonCode/mnist_train.csv'
df_train = pd.read_csv(file)
y = pd.get_dummies(df_train.iloc[:,0]).values
X = np.array(df_train.iloc[:,1:],dtype = float)

# Normalize data (min = 0., max = 1.)
X /= np.max(X)

# Split data into training and validation sets
X_train, X_validate, y_train, y_validate = cross_validation.train_test_split(X, y, test_size=0.2)

### Initialization

# Training options
warm_start = True	# Use on pretrained weights
save_params = True	# Save trained weights
track_validation = False # Track validation error
test = True
epochs = 10
batch_size = 100
epoch_size = int(X_train.shape[0]/batch_size)

# Architecture
nodes = [800,800]
k = len(nodes)

# Regularization methods
dropout_p = 0. # Dropout parameter

# SGD optimization methods
sgd_methods = ['adagrad','adadelta','rmsprop','adam']
adadelta = True
adagrad = False
rmsprop = False

if adadelta:
	rho = 0.999	# Adadelta parameter
	epsilon = 1e-6	# Adedelta parameter
elif adagrad:
	nu = 0.1
elif rmsprop:
	nu = 0.1
	rho = 0.9
else:
	nu = 0.1

# Initialize weight matrices
if warm_start:
	W = np.load('weights_%s.npy' %(str(nodes)))
else:
	uniform_bound = 1./(X.shape[0])**(0.5)
	W = [np.random.uniform(-uniform_bound,uniform_bound,(X_train.shape[1],nodes[0]))]
	for i in range(1,k):
		W.append(np.random.uniform(-uniform_bound,uniform_bound,(nodes[i-1]+1,nodes[i])))
	W.append(np.random.uniform(-uniform_bound,uniform_bound,(nodes[k-1]+1,y_train.shape[1])))

# Initialize weight dependent parameters
if adadelta: # Initialize adadelta variables	
	adadelta_g = [0. for i in range(len(W))]
	adadelta_w = [0. for i in range(len(W))]
elif adagrad:
	cache = [0. for i in range(len(W))]
elif rmsprop:
	cache = [0. for i in range(len(W))]

# Define functions
def activation(v, derivative = False):
	if not derivative:
		return 1.7159 * np.tanh(2./3.*v)
	else:
		return 1.1439 * (1. - (np.tanh(2./3. * v)) ** 2).T

def softmax(X):
    Z = np.sum(np.exp(X))
    return np.exp(X) / Z

def forward_propagate(X):
	s = activation(X.dot(W[0]))
	s = np.append(np.ones((s.shape[0],1)),s,axis = 1) # Append bias node
	Z = [s]
	for i in range(1,k):
		s = activation(Z[i-1].dot(W[i]))
		s = np.append(np.ones((s.shape[0],1)),s,axis = 1) # Append bias node
		Z.append(s)
	return softmax(Z[k-1].dot(W[k]))

def predict(X): # Convert probabilities to predictions
	a = np.argmax(forward_propagate(X),axis = 1)
	y = np.zeros((a.size,10))
	y[np.arange(a.size),a] = 1
	return y

def RMS(X,epsilon = 1e-6):
	return np.sqrt(X + epsilon)

def update_progress(progress):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

print "Initializing neural network training..."
errors = []

for epoch in range(epochs):
	print "epoch:", epoch + 1
	# Shuffle sets
	X_train, X_validate, y_train, y_validate = cross_validation.train_test_split(X, y, test_size=0.2)

	start_time = time.time()
	for i in range(epoch_size):
		update_progress((i+1.)/float(epoch_size))
		
		X_batch = X[i*batch_size:(i+1)*batch_size,:]
		y_batch = y[i*batch_size:(i+1)*batch_size,:]

		# Forward propagate and store derivatives
		Z = []
		s = activation(X_batch.dot(W[0]))
		s *= np.random.binomial(n = 1,p = 1. - dropout_p, size = s.shape) # Dropout
		s = np.append(np.ones((s.shape[0],1)),s,axis = 1) # Append bias node
		Z.append(s)
		Z_prime = [activation(X_batch.dot(W[0]),derivative = True)]

		for j in range(1,k):
			s = activation(Z[j-1].dot(W[j]))
			s *= np.random.binomial(n = 1,p = 1. - dropout_p, size = s.shape) # Dropout
			s = np.append(np.ones((s.shape[0],1)),s,axis = 1) # Append bias node
			Z.append(s)
			s_prime = activation(Z[j-1].dot(W[j]),derivative = True)
			Z_prime.append(s_prime)

		# Calculate softmax probabilities
		y_hat = softmax(Z[k-1].dot(W[k]))

		if track_validation:
			# Calculate validation error
			y_predict = predict(X_validate)
			error = 1. - accuracy_score(y_predict,y_validate)
			errors.append(error)

		# Backpropagation
		delta_hat = (y_hat - y_batch).T
		delta = [Z_prime[k-1] * W[k][1:,:].dot(delta_hat)]
		for j in range(1,k):
			delta.append(Z_prime[k-j-1] * W[k-j][1:,:].dot(delta[j-1]))

		# Compute gradients
		gradients = [(delta[k-1].dot(X_batch)).T]
		for j in range(1,k):
			gradients.append((delta[k-j-1].dot(Z[j-1])).T)
		gradients.append((delta_hat.dot(Z[k-1])).T)

		# Weight update methods
		deltaW = []
		if adadelta:
			for j in range(k):
				adadelta_g[j] = rho * adadelta_g[j] + (1. - rho) * gradients[j] ** 2
				deltaW.append(-RMS(adadelta_w[j]) * gradients[j] / RMS(adadelta_g[j]))
				adadelta_w[j] = rho * adadelta_w[j] + (1. - rho) * deltaW[j] ** 2
				W[j] += deltaW[j]

		elif adagrad:
			for j in range(k):
				cache[j] += gradients[j] ** 2
				deltaW.append(-nu * gradients[j] / RMS(cache[j]))
				W[j] += deltaW[j]

		elif rmsprop:
			for j in range(k):
				cache[j] = rho * cache[j] + (1. - rho) * gradients[j] ** 2
				deltaW.append(-nu * gradients[j] / RMS(cache[j]))
				W[j] += deltaW[j]
		else:
			for j in range(k):
				deltaW.append(-nu * gradients[j])
				W[j] += deltaW[j]

	y_predict = predict(X_validate)
	print "Accuracy:",accuracy_score(y_predict,y_validate)
	print "Time:", time.time() - start_time

print "Completed training..."

## TEST
if test:
	print "Opening test set..."
	file = '/Users/miguelbenavides/Documents/PythonCode/mnist_test.csv'
	df_test = pd.read_csv(file)
	y_test = pd.get_dummies(df_test.iloc[:,0]).values
	X_test = np.array(df_test.iloc[:,1:],dtype = float)
	X_test /= np.max(X_test)

	y_predict_test = predict(X_test)
	print "Test Accuracy:", accuracy_score(y_test,y_predict_test)

if save_params:
	# Save weight parameters
	print "Saving weight parameters..."
	np.save('weights_'+str(nodes),W)
	print "Done!"

if track_validation:
	plt.plot(np.arange(len(errors)),np.array(errors))
	plt.show()
