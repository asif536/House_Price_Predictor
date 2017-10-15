import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from function.plot import plot_Data 
from function.cost import CostFuction
from function.gradiant_desent import Gradiant_descent
from function.input_data import InputData

# reading the data 
col_name    	=   	["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors"]
data  		=	pd.read_csv('trian.csv',header=None,names=col_name).as_matrix()
x   		=	data[:,3]
y		=	data[:,0]

# ploting the data of sqft_living vs price
  
plot_Data(x,y)

# Main Function
def LinearRegression():
	X 				=	data[:,1:6]
	Y 				=       data[:,0]
	m 				= 	len(Y)
	Y 				=	Y.reshape(m,1)
	X 				= 	np.c_[np.ones(m),X] # Adding column of 1 of size m in X
	theta				= 	np.zeros((6,1)) 	# initailzing theta value
	learning_rate			= 	0.000000001
	iterations			=	1200
	J				=	CostFuction(X,Y,m,theta)
	print " Testing Initial Value of Cost Function %f"%J
	[theta,j_history]		=	Gradiant_descent(X,Y,theta,m,learning_rate,iterations)
	
	#print j_history

	#print theta

	# Predicting price for input dataset
	test_data1	=	[1,4,4.5,5420,101930,1]
	test_data1	= 	np.asarray(test_data1)
	Predict1 	=	test_data1.dot(theta)
	print "Estimate Price of House is: $%f "%Predict1
	print "Original Price : $1225000 "

	test_data2	=	[1,3,2,1370,9680,1]
	test_data2	= 	np.asarray(test_data2)
	Predict2 	=	test_data2.dot(theta)
	print "Estimate Price of House is: $%f "%Predict2
	print "Original Price : $400000 "

	test_data3	=	[1,5,2,1810,4850,1.5]
	test_data3	= 	np.asarray(test_data3)
	Predict3 	=	test_data3.dot(theta)
	print "Estimate Price of House is: $%f "%Predict3
	print "Original Price : $530000 "

	Output          =  InputData(theta)
	print "Estimate Price of House is: $%f"%Output



if __name__ == '__main__':
	LinearRegression()
