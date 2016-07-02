# ANN on time series data
import datetime
import numpy as np
import matplotlib.colors as colors
import matplotlib.finance as finance
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

startdate = datetime.date(2002, 1, 1)
today = enddate = datetime.date.today()
ticker = '^GSPC'

# a numpy record array with fields: date, open, high, low, close, volume, adj_close)
fh = finance.fetch_historical_yahoo(ticker, startdate, enddate)
r = mlab.csv2rec(fh)
fh.close()
r.sort()
view_length = 3000

plt.plot(r.date[-view_length:],r.close[-view_length:])
plt.show()
# Transform time series with first difference and max normalize

r_delta = []
for i in range(1,len(r)):
	r_delta.append(r.close[i] - r.close[i-1])
scale = max(r_delta)
r_delta = np.array(r_delta) / max(r_delta)
r_delta = r_delta.tolist()

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

# Parse into training and test data sets
r_parse = 60

X_train = X[:(length-r_parse),:]
X_test = X[(length - r_parse):,:]
y_train = y[:(length-r_parse),:]
y_test = y[(length-r_parse):,:]

# Create minibatches; determine batch size
r_batch = 0.3
index_batch = np.random.choice([True,False],X_train.shape[0],p=[r_batch,1-r_batch])
X_batch =  X_train[index_batch]
y_batch = y_train[index_batch]

# Build neural network

# Set hyperparameters, k as the number of hidden layers
k = 7

nodes = [120,64,32,16,8,4,2]
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

while iteration <= 1000:
	# Create minibatches; determine batch size
	r_batch = 0.1
	index_batch = np.random.choice([True,False],X_train.shape[0],p=[r_batch,1-r_batch])
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

	# Compute errors
	error = 0.5 * (y_batch - y_hat).T.dot(y_batch - y_hat)
	iteration += 1
	print "Iteration:", iteration,"in-error:",error[0,0],"out-error:",error_test[0,0]
	errors.append(error[0,0])


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

# Reconstruct forecast
def feedforward(X):
	temp = np.tanh(X.dot(W[0]))
	temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
	y_predict = temp

	for i in range(1,k):
		temp = np.tanh(y_predict.dot(W[i]))
		temp = np.append(np.ones((temp.shape[0],1)),temp,axis = 1) # Append bias node
		y_predict = temp
	y_predict = np.tanh(y_predict.dot(W[k]))

	return y_predict[0,0]

x_predict = np.array([r_delta[-window_length:]])

r_close_list = r.close.tolist()
r_date_list = r.date.tolist()
s = 365*2
for i in range(s):
	x_predict = np.array([r_delta[-window_length:]])
	r_delta_predict = feedforward(x_predict)
	r_delta.append(r_delta_predict)
	r_close_list.append((r_close_list[-1]+scale*r_delta_predict))
	r_date_list.append(datetime.date.today()+datetime.timedelta(days=i))

view_length = 365*4
r_close = np.array(r_close_list[-view_length:])
r_date = np.array(r_date_list[-view_length:])

plt.plot(r_date,r_close)
plt.plot(r.date[-view_length:],r.close[-view_length:])
plt.title('Artificial Neural Network Forecast: %s'%(ticker))
plt.xlabel('Date')
plt.show()
