import numpy as np
from cost import CostFuction


# Calculating gradiant descent and minimize the value of theta

def Gradiant_descent(x,y,theta,m,learning_rate,num_iters):
	j_history = np.zeros((num_iters,1))
	j_history = j_history.reshape(num_iters,1)
	for i in range(num_iters):
		hypothysis		=	x.dot(theta)
		diff			=	(hypothysis-y)
		gradiant 		=	(x.T).dot(diff)/m
		theta			=	theta-(learning_rate*gradiant)
		j_history[i]	=	CostFuction(x,y,m,theta)
		print j_history[i]
	return [theta,j_history]

