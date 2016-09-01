# NaiveBayes.py
# Implementation of naive Bayes algorithm from scratch
# Accuracy ~ 75%

import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import cross_validation

raw_data = pd.read_csv('GermanCredit.csv')
raw_data = raw_data.dropna(subset=["credit_history", "purpose","savings_status", "employment", "property_magnitude"])
y = raw_data.iloc[:,-1]
X = raw_data.iloc[:,:-1]

categorical_indexes = [0,2,3,5,6,7,8,9,11,13,14,16,18,19]
numerical_indexes = [1,4,10,12,15,17]

categorical_names = list(raw_data.ix[:,categorical_indexes].columns)
numerical_names = list(raw_data.ix[:,numerical_indexes].columns)

# Vectorize categoricals
x_cat_vector = pd.get_dummies(raw_data.iloc[:,categorical_indexes[0]], prefix = categorical_names[0])

for i in range(1,len(categorical_indexes)):
	x_cat_vector = pd.concat([x_cat_vector,pd.get_dummies(raw_data.iloc[:,categorical_indexes[i]],prefix = categorical_names[i])],axis = 1)

x_num = pd.get_dummies(raw_data.iloc[:,numerical_indexes])

X = pd.concat([x_cat_vector,x_num], axis = 1)
y = pd.get_dummies(y).iloc[:,0]

### BUILD MODEL
for k in range(50):

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

	## Construct prior tables
	y_good = y_train.loc[y_train.loc[y_train == 0.].index]
	y_good.name = 'good'
	y_bad = y_train.loc[y_train.loc[y_train == 1.].index]

	n_total = X_train.loc[y_good.index].shape[0] + X_train.loc[y_bad.index].shape[0]

	n_good_total = X_train.loc[y_good.index].shape[0]
	n_bad_total = X_train.loc[y_bad.index].shape[0]

	p_good = float(n_good_total) / float(n_total)
	p_bad = 1. - p_good

	# Calculate categorical priors
	p_good_priors = []
	p_bad_priors = []
	for i in range(X_train.shape[1]-6):
		p_good_priors.append([X_train.ix[y_good.index,i].name, 
			X_train.ix[y_good.index,i].sum() / n_good_total])
		p_bad_priors.append([X_train.ix[y_bad.index,i].name, 
			X_train.ix[y_bad.index,i].sum() / n_bad_total])
	p_good_priors = dict(p_good_priors)
	p_bad_priors = dict(p_bad_priors)

	p_good_priors_gauss = []
	p_bad_priors_gauss = []
	# Calculate numerical priors, assume Gaussian, estimate parameters (mean, variance)

	for i in range(X_train.shape[1]-6,X_train.shape[1]):
		p_good_priors_gauss.append([X_train.ix[y_good.index,i].name,
			[X_train.ix[y_good.index,i].mean(),
			X_train.ix[y_good.index,i].std()]])
		p_bad_priors_gauss.append([X_train.ix[y_bad.index,i].name,
			[X_train.ix[y_bad.index,i].mean(),
			X_train.ix[y_bad.index,i].std()]])

	p_good_priors_gauss = dict(p_good_priors_gauss)
	p_bad_priors_gauss = dict(p_bad_priors_gauss)

	# Predict
	y_good_test = y_test.loc[y_test.loc[y_test == 0.].index]
	y_good_test.name = 'good'
	y_bad_test = y_test.loc[y_test.loc[y_test == 1.].index]

	count = 0
	for j in range(X_test.shape[0]):
		x_profile = X_test.iloc[j,:-6]
		x_profile = x_profile.loc[x_profile == 1.]

		p_good = 1.
		p_bad = 1.

		# Calculate posteriors for categoricals
		for i in range(len(list(x_profile.index))):
			p_good *= p_good_priors[x_profile.index[i]]
		for i in range(len(list(x_profile.index))):
			p_bad *= p_bad_priors[x_profile.index[i]]

		# Calculate posteriors for numerics
		x_profile = X_test.iloc[j,-6:]

		for i in range(len(list(x_profile.index))):
			x = x_profile.index[i]
			p_good *= stats.norm.pdf(x_profile.iloc[i],loc = p_good_priors_gauss[x][0], 
				scale = p_good_priors_gauss[x][1])
			p_bad *= stats.norm.pdf(x_profile.iloc[i],loc = p_bad_priors_gauss[x][0], 
				scale = p_bad_priors_gauss[x][1])

		# Decision rule
		if p_good < p_bad:
			y_predict = 1.
		else:
			y_predict = 0.

		# Count accuracy
		if y_predict == y_test.iloc[j]:
			count += 1

	print k+1,"accuracy:",float(count) / float(X_test.shape[0])