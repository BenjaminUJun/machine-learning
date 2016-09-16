### Polynomial ridge regression demonstration
#
# Implementation of polynomial ridge regression with linear algebra
# 
#
# Created by Miguel Benavides on September 15, 2016
#
# GitHub: https://github.com/MiguelBenavides/machine-learning
# Email: migibenavides@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cross_validation
import time, sys

# Choose data to demonstrate
UKdata = True
Damp_Exponential = False
Cubic_Polynomial = False

if UKdata:
	file = '/Users/miguelbenavides/Downloads/EmplUK.csv'	# Change file path to own directory
	df = pd.read_csv(file)
	X = df['emp'].values
	y = df['capital'].values
	N = X.shape[0]

	# Define bounds of regularization terms lambda: higher lambda corresponds to higher regularization
	upper_bound = 100.
	lower_bound = 0	# First curve starts with no regularization
	regularization_terms = 20	# number of regularized curves to fit
	lambdas = np.arange(lower_bound,upper_bound,upper_bound/regularization_terms)
 
elif Damp_Exponential:	# Higher degree better for approximating damp exponential (cf. Taylor series approximations)
	N = 200	# Number of points
	X = np.e*np.random.random(N)	# Distribution of x coordinates (chosen here for visual convenience)
	
	def damp_exponential(X):
		y = np.exp(-X) * np.cos(2*np.pi*X) 	# User-defined function plus Gaussian noise
		return y
	y = damp_exponential(X)+ 0.1*np.random.randn(N) # Add noise

	upper_bound = 5e-4
	lower_bound = 0
	regularization_terms = 20
	lambdas = np.arange(lower_bound,upper_bound,upper_bound/regularization_terms)

elif Cubic_Polynomial:	# Demonstrate overfitting on cubic equation
	N = 100
	X = 15*np.random.random(N)-7.5 # Defines range of function
	
	def cubic(X):
		y = (2 * X ** 3) -  (X ** 2) + (5 * X)
		return y
	y = cubic(X) + 150*np.random.randn(N) # Add noise

	upper_bound = 1.
	lower_bound = 0	
	regularization_terms = 10
	lambdas = np.arange(lower_bound,upper_bound,upper_bound/regularization_terms)

# Split into training and testing sets
test_size = 0.3
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size)
text_x = np.min(X)
text_y = np.max(y) + 1


# Define range of polynomial degrees k to fit (k=1 fits a horizontal line, k=2 is linear regression)
k_lower = 2
k_upper = k_lower + 36	# Plots nine degrees per figure
k_range = float(k_upper - k_lower)
num_fig = int(k_range/9.)
plt.style.use('ggplot')

R_squared_df = []

fig = 1
for k in range(k_lower,k_upper):

	# Cycle through figures
	if (k-k_lower) % 9 == 0:
		plt.figure(fig)
		fig += 1

	# Create subplots and align on grid
	ax = plt.subplot(3,3,(k-k_lower) % 9 + 1)
	plt.plot(X_train,y_train,'o',color = '#034f84')	# Blue for training set
	plt.plot(X_test,y_test,'o',color = '#fae03c')	# Yellow for test set
	
	# Plot true line generating artificial data
	if Cubic_Polynomial:	
		x_range = np.arange(np.min(X),np.max(X), (np.max(X)-np.min(X)) / 100.)
		plt.plot(x_range,cubic(x_range),'--', color = '#FC4C02')
	elif Damp_Exponential:
		x_range = np.arange(np.min(X),np.max(X), (np.max(X)-np.min(X)) / 100.)
		plt.plot(x_range,damp_exponential(x_range),'--',color = '#FC4C02')

	# Set window view size
	plt.xlim(np.min(X),np.max(X))
	plt.ylim(np.min(y),np.max(y))
	plt.title('degree:%r'%(k-1),fontsize = 12)

	R_squared_list = []
	for i in range(lambdas.size):
		# TRAINING: Construct Vandermonde matrix and estimate polynomial fit coefficients beta
		X_vand_train = np.vander(X_train,k)
		beta = np.linalg.inv(X_vand_train.T.dot(X_vand_train)+lambdas[i]*np.identity(k)).dot(X_vand_train.T).dot(y_train)

		# TEST: Compute R Squared on test set
		X_vand_test = np.vander(X_test,k)
		H = X_vand_test.dot(np.linalg.inv(X_vand_test.T.dot(X_vand_test)+lambdas[i]*np.identity(k))).dot(X_vand_test.T)
		l = np.ones(X_vand_test.shape[0]).reshape((X_vand_test.shape[0]),1)
		M = l.dot(np.linalg.inv(l.T.dot(l))).dot(l.T)
		Q = (np.identity(H.shape[0]) - H).T.dot(np.identity(H.shape[0]) - H)
		P = (np.identity(M.shape[0]) - M).T.dot(np.identity(M.shape[0]) - M)
		R_squared = 1. - (y_test.T.dot(Q).dot(y_test)) / (y_test.T.dot(P).dot(y_test))
		R_squared_list.append(R_squared)
		R_squared_df.append([k,i,R_squared])

		# Plot polynomial ridge curve
		a = np.arange(np.min(X_train),np.max(X_train),0.1)
		a_vand = np.vander(a,k)
		plt.plot(a,a_vand.dot(beta),color = '#dd4132',alpha = 1. - float(lambdas[i]) / upper_bound) # More transparent lines mean stronger regularization
		plt.axis('off')
	
	# Annotate mean R squared statistic (averaged over all regularized curves)
	R_squared_list = np.array(R_squared_list)
	ax.text(text_x,text_y,"Mean $R^2$:%r"%(round(np.mean(R_squared),3)), fontsize =10)

# Plot R Squared statistic curves over degrees
plt.figure(fig+1)
R_squared_df = pd.DataFrame(R_squared_df)
for i in range(lambdas.size):
	if i == 0:
		z = R_squared_df.iloc[:,2][R_squared_df[1] == i].values
		plt.plot(np.arange(z.size),z,color = '#034f84')
	else:
		z = R_squared_df.iloc[:,2][R_squared_df[1] == i].values
		plt.plot(np.arange(z.size),z,color = '#dd4132',alpha = 1. - float(lambdas[i]) / upper_bound)
plt.ylim((0,1))
plt.title('Effect of Degree on $R^2$')
plt.xlabel('Degree')
plt.ylabel('$R^2$')
plt.show()