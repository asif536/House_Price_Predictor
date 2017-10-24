import numpy as np 


def FeatureScaling(x):
	for i in range(1,7):
		mean 	= np.mean(x[:,i])
		std	= np.std(x[:,i])
		x[:,i]	= (x[:,i]-mean)/std

	return x

def FeatureScalingInput(input_x,X):
	for i in range(1,7):
		input_x[i] = float((input_x[i]-np.mean(X[:,i]))/np.std(X[:,i]))
	return input_x
