import numpy as np

def accuracy(theta):
	x=raw_input("Input the features\n").split(',')
	p=np.asarray(x)
	a=map(float,p)
	predict=np.dot(a,theta)
	return predict
	
	
	