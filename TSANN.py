#
#		Forecasting time series data with neural networks
#
#		TSANN.py
#
# Implementation of artificial neural network to forecast financial time series data
# Uses financial data from Yahoo! Finance
# 
# Features v1:
# - Divided into three phases: Initialization, Training, and Testing
# - Initialization opens MNIST training data and preprocesses for training
# - Training involves the core neural network algorithm
# - Testing opens MNIST test data, applies trained model, and evaluates performance
# - Added feature to save model parameters if performance exceeds 96% on test data
# - Changed activation function to 1.7519 tanh(2/3z) in line with LeCun's Efficient Backprop guidelines
# - Added dropout procedure
# 
# Features v2:
# - Steamlined general flow into preprocessing, initialization, and testing
# - Defined recurring functions for conciseness
# - Implemented various stochastic gradient descent optimization methods, vastly improving convergence rate
# - Added progress bar when training per epoch
# - Fixed last activation layer to use softmax
# - Added 'warm start' feature to use pretrained weights from saved sessions
# - Added feature to determine batch size and cross validation split size
#
# Created by Miguel Benavides on September 16, 2016
#
# GitHub: https://github.com/MiguelBenavides/machine-learning
# Email: migibenavides@gmail.com

import datetime
import numpy as np
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import time, sys
from sklearn import cross_validation

startdate = datetime.date(2012, 1, 1)
today = enddate = datetime.date.today()
ticker = '^DJI'

# a numpy record array with fields: date, open, high, low, close, volume, adj_close)
fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)
r = mlab.csv2rec(fh)
fh.close()
r.sort()
view_length = 3000

# Plot data
plt.style.use('ggplot')
plt.figure(1)
plt.subplot(211)
plt.title(ticker)
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.plot(r.date[-view_length:],r.close[-view_length:])

# Transform time series to difference returns
r_delta = []
for i in range(1,len(r)):
	r_delta.append(r.close[i] - r.close[i-1])
scale = max(r_delta)
r_delta = np.array(r_delta) / scale
r_delta = r_delta.tolist()

# Plot data
plt.subplot(212)
plt.title("Difference returns: %s"%(ticker))
plt.xlabel('Date')
plt.plot(r.date[1:],r_delta)
plt.show()

window_length = 180
period_length = len(r)

# X is window_length - 1 long with the last being the target value y
X = []
y = []
for i in range(period_length - window_length):
	X.append(r_delta[i:i+window_length-1])
	y.append(r_delta[i+window_length-1])

X = np.array(X)
y = np.array(y).reshape((len(y),1))

length = X.shape[0]

X = np.concatenate((np.ones((length,1)),X),axis = 1)

test_size = 0.1
# Parse into training and test data sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size)

### Initialization

# Training options
warm_start = False	# Use on pretrained weights
save_params = False	# Save trained weights
track_error = True # Track validation error
test = True
epochs = 100
batch_size = 10
epoch_size = int(X_train.shape[0]/batch_size)

# Architecture
nodes = [300,300]
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
	return Z[k-1].dot(W[k])

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
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size)

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

		# Calculate y hat
		y_hat = Z[k-1].dot(W[k])

		if track_error:
			# Calculate error
			y_predict = forward_propagate(X_test)
			error = (y_predict - y_test)
			error = error.T.dot(error)[0][0]
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

	y_predict = forward_propagate(X_test)
	error = (y_predict - y_test)
	error = error.T.dot(error)[0][0]
	print "Error:",error
	print "Time:", time.time() - start_time

print "Completed training..."

x_predict = np.array([r_delta[-window_length:]])

r_close_list = r.close.tolist()
r_date_list = r.date.tolist()

s = 720
for i in range(s):
	x_predict = np.array(r_delta[-window_length:])
	x_predict = x_predict.reshape((x_predict.shape[0],1)).T
	r_delta_predict = forward_propagate(x_predict)[0][0]
	r_delta.append(r_delta_predict)
	r_close_predict = r_close_list[-1] + scale * r_delta_predict
	r_close_list.append(r_close_predict)
	r_date_list.append(datetime.date.today()+datetime.timedelta(days=i))

view_length = s
r_close = np.array(r_close_list[-view_length:])
r_date = np.array(r_date_list[-view_length:])

# Plot errors
errors = np.array(errors)
plt.figure(2)
plt.subplot(211)
l1, = plt.plot(np.arange(errors.size),errors)
plt.title('Errors')
plt.yscale('log')
plt.xlabel('Iterations')

# Plot forecast
plt.subplot(212)
l3, = plt.plot(r_date,r_close)
l4, = plt.plot(r.date[-view_length:],r.close[-view_length:])
plt.title('Artificial Neural Network Forecast: %s'%(ticker))
plt.xlabel('Date')
plt.legend((l3,l4),('forecast','historical'),loc = 'upper right')
plt.show()
