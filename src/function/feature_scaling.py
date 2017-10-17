import numpy as np 


def FeatureScaling(x):
	for i in range(1,7):
		mean 	= np.mean(x[:,i])
		std		= np.std(x[:,i])
		x[:,i]	= (x[:,i]-mean)/std

	return x
