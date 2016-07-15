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
# Notes on performance: 
# - Dependencies include numpy and matplotlib
# - Achieves ~ 98% classification rate on a [375,75] architecture over 2000 iterations in 6 minutes
# - Requires about 2 GB or RAM
# 
# Notes on structure:
# - Divided into three phases: Initialization, Training, and Testing
# - Initialization opens MNIST training data and preprocesses for training
# - Training involves the core neural network algorithm
# - Testing opens MNIST test data, applies trained model, and evaluates performance
# - Added feature to save model parameters if performance exceeds 96% on test data
#
# Features to add for improvement:
# - Classification evaluation algorithm can be further optimized
# - Neuro-evolution for optimal architecture
# - Weight decay
# - Adaptive learning rate with gradients
# - GPU implementation
# 
# Current performance: 98.02%
# Goal: 99% classification rate


# Created by Miguel Benavides on July 15, 2016
#
# GitHub: https://github.com/MiguelBenavides/machine-learning
# Email: migibenavides@gmail.com

import csv
import numpy as np
import matplotlib.pyplot as plt
import time

print "Loading MNIST training set..."
print "-----------------------------"

### INITIALIZATION STAGE

# Open MNIST training data set
raw_data_train = []
f_train = csv.reader(open('mnist_train.csv','rb'))
for row in f_train:
	raw_data_train.append(row)

# Convert labels into vectors for both sets
n_train = len(raw_data_train) # Size of data set
print "Training size:", n_train
label_train = []
data_train = []
for i in range(n_train):
	v = np.zeros(10)
	label_number = raw_data_train[i][0]
	v[label_number] = 1.
	label_train.append(v)
	data_train.append(raw_data_train[i][1:])

# Define functions
def show_digit(index): # For testing purposes
	rdata = np.array(data_train[index],dtype = float).reshape((28,28))
	plt.matshow(rdata,cmap = plt.cm.gray)
	plt.show()

def hardmax(v): # Used in classification
	for i in range(v.shape[0]):
		if v[i] < v.max():
			v[i] = 0.
		else:
			v[i] = 1.
	return v

# Construct and normalize data
X_full_train = np.array(data_train,dtype= float) / 255. # Normalize by maximum pixel intensity value
X_full_train -= np.mean(X_full_train,axis = 0) # Mean center
y_full_train = np.array(label_train,dtype= float)

# Append bias nodes
X_full_train = np.concatenate((np.ones((n_train,1)),X_full_train),axis = 1)

### TRAINING STAGE

# Parse data into training and validation sets
r = 0.9
index_train = np.random.choice([True,False],n_train,p=[r,1-r])
index_validate = ~ index_train

X_train = X_full_train[index_train]
X_validate = X_full_train[index_validate]
y_train = y_full_train[index_train]
y_validate = y_full_train[index_validate]

print "Number of training examples:", X_train.shape[0]
print "Number of validation examples:", X_validate.shape[0]
print "-----------------------------"

# Build neural network

# Set hyperparameters, k as the number of hidden layers
nodes = [400,75] # Neural network architecture [x1,x2,..] where xi is the number of nodes in layer i
k = len(nodes)
nu = 1.5 # learning rate
gamma = 0.001 # adaptive learning rate hyperparameter / nu decay
print "Setting hyperparameters..."
print "Number of layers:",k
print "Architecture:", nodes
print "Learning rate:", nu
print "-----------------------------"

# Create initial minibatches
r_batch = 0.01
index_batch = np.random.choice([True,False],X_train.shape[0],p=[r_batch,1-r_batch])
X_batch =  X_train[index_batch]
y_batch = y_train[index_batch]

# Construct weight matrices 

initial_normal = False
initial_uniform = True

if initial_normal == True:
	W_size = 0.005 # scales down variance of initial weights
	W = [W_size * np.random.randn(X_batch.shape[1],nodes[0])]
	for i in range(1,k):
		W.append(W_size * np.random.randn(nodes[i-1]+1,nodes[i]))
	W.append(W_size * np.random.randn(nodes[k-1]+1,y_batch.shape[1]))

elif initial_uniform == True:
	uniform_bound = 1./(n_train)**(0.5)
	W = [np.random.uniform(-uniform_bound,uniform_bound,(X_batch.shape[1],nodes[0]))]
	for i in range(1,k):
		W.append(np.random.uniform(-uniform_bound,uniform_bound,(nodes[i-1]+1,nodes[i])))
	W.append(np.random.uniform(-uniform_bound,uniform_bound,(nodes[k-1]+1,y_batch.shape[1])))

# Initialize main loop
iteration = 0
performance = 0.
errors = []
errors_validation = []
errors_classification = []

# Iteration options (user defined)
track_validation = True
track_classification = False
nu_decay = True

print "Initializing neural network training..."

# Define forward propagation
def forward_propagate(X):
	s = np.tanh(X.dot(W[0]))
	s = np.append(np.ones((s.shape[0],1)),s,axis = 1) # Append bias node
	Z = [s]
	for i in range(1,k):
		s = np.tanh(Z[i-1].dot(W[i]))
		s = np.append(np.ones((s.shape[0],1)),s,axis = 1) # Append bias node
		Z.append(s)
	return np.tanh(Z[k-1].dot(W[k]))

# Begin timing main loop
start_time = time.time()
iteration_limit = 2000

while iteration <= iteration_limit:

	# Create minibatches
	index_batch = np.random.choice([True,False],X_train.shape[0],p=[r_batch,1-r_batch])
	X_batch =  X_train[index_batch]
	y_batch = y_train[index_batch]

	# Feedforward Z's and Z_prime's
	temp = np.tanh(X_batch.dot(W[0]))
	temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
	Z = [temp]
	Z_prime = Z_prime = [(1. - (np.tanh(X_batch.dot(W[0]))) ** 2).T]

	for i in range(1,k):
		temp = np.tanh(Z[i-1].dot(W[i]))
		temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
		Z.append(temp)
		temp = (1.-(np.tanh(Z[i-1].dot(W[i]))) ** 2).T
		Z_prime.append(temp)

	y_hat = np.tanh(Z[k-1].dot(W[k]))

	# Backpropagation
	delta_hat = (y_hat - y_batch).T
	delta = [Z_prime[k-1] * W[k][1:,:].dot(delta_hat)]
	for i in range(1,k):
		delta.append(Z_prime[k-i-1] * W[k-i][1:,:].dot(delta[i-1]))

	# Adapt learning rate
	if nu_decay == True:
		nu_adapt = nu / (1.+iteration*gamma)**2
	else:
		nu_adapt = nu

	# Update weights
	deltaW = [-nu_adapt * (delta[k-1].dot(X_batch)).T]
	for i in range(1,k):
		deltaW.append(-nu_adapt * (delta[k-i-1].dot(Z[i-1])).T)
	deltaW.append(-nu_adapt * (delta_hat.dot(Z[k-1])).T)

	for i in range(k):
		W[i] += deltaW[i]

	# In-sample error
	error = 0.5 * (y_batch - y_hat).T.dot(y_batch - y_hat)
	errors.append(error[0,0])

	iteration += 1

	# Track validation error
	if track_validation == True:
		# Forward propagate using validation data
		y_validate_hat = forward_propagate(X_validate)
		error_validate = 0.5 * (y_validate_hat - y_validate).T.dot(y_validate_hat - y_validate)
		errors_validation.append(error_validate[0,0])

		# Display progress and classification rate every 5%
		percentage_complete = (float(iteration) / float(iteration_limit)) * 100.

		if percentage_complete % 5 == 0:
			print "Training...%r" %(percentage_complete),"%"
			# Classification error
			y_validate_hat_predict = np.apply_along_axis(hardmax,1,y_validate_hat)
			y_model_predict = y_validate - y_validate_hat_predict

			count = 0.
			m = y_model_predict.shape[0]

			for i in range(m):
				if all(y_model_predict[i]==0.):
					count += 1.
			performance = (count / m) * 100.
			errors_classification.append(performance)
			print iteration, "Classification rate:", performance,"%","(%i / %i)" %(count,m)
	
	# Display classification rate per epoch
	if track_classification == True:
		# Forward propagate using validation data
		y_validate_hat = forward_propagate(X_validate)
		y_validate_hat_predict = np.apply_along_axis(hardmax,1,y_validate_hat)
		y_model_predict = y_validate - y_validate_hat_predict

		count = 0.
		m = y_model_predict.shape[0]

		for i in range(m):
			if all(y_model_predict[i]==0.):
				count += 1.
		performance = (count / m) * 100.
		errors_classification.append(performance)
		print iteration, "Classification rate:", performance,"%","(%i / %i)" %(count,m)

print "-----------------------------"
print "Completed training..."
model_time = time.time() - start_time
print "Training time:", model_time, "seconds"
print "Neural network architecture: ", nodes


# Classification error
y_validate_hat_predict = np.apply_along_axis(hardmax,1,y_validate_hat)
y_model_predict_training = y_validate - y_validate_hat_predict

count = 0.
for i in range(y_model_predict_training.shape[0]):
	if all(y_model_predict_training[i]==0.):
		count += 1.
print "Number of training examples:", X_train.shape[0]

print "Correctly classified", int(count)
print "Number of validation examples:", y_model_predict_training.shape[0]
print "Validation classification rate:", (count / y_model_predict_training.shape[0]) * 100.,"%"

display_plot = False

if display_plot == True:
# Plot errors
	if track_validation == True:
		errors = np.array(errors)
		plt.style.use('ggplot')
		l1, = plt.plot(np.arange(iteration),errors)
		l2, = plt.plot(np.arange(iteration),errors_validation)
		plt.title('Errors')
		plt.yscale('log')
		plt.xlabel('Iterations')
		plt.legend((l1,l2),('in-sample error','out-of-sample error'),loc = 'upper right')
		plt.show()

	if track_classification == True:
		errors = np.array(errors_classification)
		plt.style.use('ggplot')
		plt.title('ANN MNIST Classification Performance: Validation')
		plt.xlabel('Iterations')
		plt.plot(np.arange(iteration),errors)
		plt.show()

### TESTING STAGE

print "Loading MNIST test set..."
# Open MNIST testing data set
raw_data_test = []
f_test = csv.reader(open('mnist_test.csv','rb'))
for row in f_test:
	raw_data_test.append(row)

# Convert labels into vectors for both sets
n_test = len(raw_data_test)
label_test = []
data_test = []
for i in range(n_test):
	v = np.zeros(10)
	label_number = raw_data_test[i][0]
	v[label_number] = 1.
	label_test.append(v)
	data_test.append(raw_data_test[i][1:])

# Construct and normalize data
X_full_test = np.array(data_test,dtype = float) / 255.
X_full_test -= np.mean(X_full_test,axis =0)
y_full_test = np.array(label_test,dtype = float)

# Append bias nodes
X_full_test = np.concatenate((np.ones((n_test,1)),X_full_test),axis = 1)

# Run neural network through test data
temp = np.tanh(X_full_test.dot(W[0]))
temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
y_test_hat = temp

for i in range(1,k):
	temp = np.tanh(y_test_hat.dot(W[i]))
	temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
	y_test_hat = temp
y_test_hat = np.tanh(y_test_hat.dot(W[k]))

y_test_hat_predict = np.apply_along_axis(hardmax,1,y_test_hat)
y_model_predict = y_full_test - y_test_hat_predict

count = 0.
for i in range(y_model_predict.shape[0]):
	if all(y_model_predict[i]==0.):
		count += 1.
print "Correctly classified:", int(count)
print "Number of testing examples:", y_model_predict.shape[0]
model_performance = (count / y_model_predict.shape[0]) * 100.
print "Final test classification rate:", model_performance, "%"

# Save model if performance > 98% 
if model_performance > 98.:
	print "Saving model parameters..."
	model = open("mnist_model_parameters.txt","w")
	model.write("MNIST Neural Network Model Parameters\n")
	model.write("-------------------------------------\n")
	model.write("Neural network architecture: %s\n" %(nodes))
	model.write("Classification Performance: %f\n" %(model_performance))
	model.write("Iterations: %r\n" %(iteration))
	model.write("Computation time: %r\n"%(model_time))
	model.write("Model hyperparameters:\n")
	model.write("nu: %r\n" %(nu))
	model.write("gamma: %r\n" %(gamma))
	model.write("Model weights:\n")
	for i in range(len(W)):
		model.write("W[%i]:\n%r\n" %(i,W[i].tolist()))
	model.close()

