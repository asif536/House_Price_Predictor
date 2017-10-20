import numpy as np

# Compute the Cost function
def CostFuction(X,Y,m,theta,lamda):
	j=0.0
	hypothysis		= 	X.dot(theta)
	sqr_diff		=	np.power((hypothysis-Y),2)
	j			=	(1.0/(2*m))*np.sum(sqr_diff)+(1.0/(2*m))*lamda*np.sum(theta**2)
	return j
