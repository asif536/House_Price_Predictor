import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from function.plot import plot_Data 
from function.cost import CostFuction
from function.gradiant_desent import Gradiant_descent
from function.input_data import InputData
from function.feature_scaling import FeatureScaling,FeatureScalingInput

# reading the data 
col_name    	=   ["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","zipcode"]
data  		=	pd.read_csv('train.csv',header=None,names=col_name).as_matrix()
x   		=	data[:,3]
y		=	data[:,0]

# ploting the data of sqft_living vs price
  
#plot_Data(x,y)

# Main Function
def LinearRegression():
	X 				=	data[:,1:7]
	Y 				=   	data[:,0]
	m 				= 	len(Y)
	Y 				=	Y.reshape(m,1)
	X 				= 	np.c_[np.ones(m),X] # Adding column of 1 of size m in X
	X 				=   	FeatureScaling(X)
	theta				= 	np.zeros((7,1)) 	# initailzing theta value
	learning_rate			= 	0.1
	iterations			=	600
	lamda 				=   10
	J				=	CostFuction(X,Y,m,theta,lamda)
	#print " Testing Initial Value of Cost Function %f"%J
	[theta,j_history]		=	Gradiant_descent(X,Y,theta,m,learning_rate,iterations,lamda)
	
	# trian cost function
	J_train  			=  	CostFuction(X,Y,m,theta,lamda)

	#print " Testing Train Value of Cost Function %f"%J_train
	#test Gradiant desent

	test_data      			=	pd.read_csv("test.csv").as_matrix()
	x_test 				=	test_data[:,1:7]
	y_test 				=   	test_data[:,0]
	m_test              		=   	len(y_test)
	x_test 				= 	np.c_[np.ones(m_test),x_test]
	x_test				=   	FeatureScaling(x_test)
	j_test				=	CostFuction(x_test,y_test,m_test,theta,lamda)
	
	#print " Testing Test Value of Cost Function %f"%j_test

	#print theta


	# Predicting price for input dataset
	test_data1	=	[1.,-0.4233863,-0.35174302,-0.14527205,-0.12313223,-1.1672141,-1.34630666]
	test_data1	= 	np.asarray(test_data1)
	Predict1 	=	test_data1.dot(theta)
	print "Estimate Price of House is: $%f "%Predict1
	print "Original Price : $507500 "

	test_data2	=	[1,-0.3877495,-0.409448634,-1.0534574,-0.13330988,-0.84867380,0.256399555]
	test_data2	= 	np.asarray(test_data2)
	Predict2 	=	test_data2.dot(theta)
	print "Estimate Price of House is: $%f "%Predict2
	print "Original Price : $224000 "

	test_data3	=	[ 1.,0.68083685,1.2407209,1.93493738,0.02416507,0.54143122,-0.35520745]
	test_data3	= 	np.asarray(test_data3)
	Predict3 	=	test_data3.dot(theta)
	print "Estimate Price of House is: $%f "%Predict3
	print "Original Price : $972000"

	#Predicting price based on input
	
	x_input         	=  	InputData()
	X_train 		=	data[:,1:7]
	X_train 		= 	np.c_[np.ones(m),X_train]
	x_input			=	FeatureScalingInput(x_input,X_train)
	Predict 		=	(x_input).dot(theta)
	print "Estimate Price of House is: $%f"%Predict



if __name__ == '__main__':
	LinearRegression()
