import numpy as np

# Compute the Cost function
def CostFuction(X,Y,m,theta):
	j=0.0
	hypothysis		= 	X.dot(theta)
	sqr_diff		=	np.power((hypothysis-Y),2)
	j				=	(1.0/(2*m))*np.sum(sqr_diff)  
	return j