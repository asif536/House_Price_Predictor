import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from function.plot import plot_Data 
from function.cost import CostFuction
from function.gradiant_desent import Gradiant_descent
from function.input_data import InputData
from function.feature_scaling import FeatureScaling

# reading the data 
col_name    	=   	["price","bedrooms","bathrooms","sqft_living","sqft_lot","floors","zipcode"]
data  		=	pd.read_csv('train.csv',header=None,names=col_name).as_matrix()
x   		=	data[:,3]
y		=	data[:,0]

# ploting the data of sqft_living vs price
  
#plot_Data(x,y)

# Main Function
def LinearRegression():
<<<<<<< HEAD
	X 				=	data[:,1:7]
	Y 				=   data[:,0]
	m 				= 	len(Y)
	Y 				=	Y.reshape(m,1)
	X 				= 	np.c_[np.ones(m),X] # Adding column of 1 of size m in X
	X 				=   	FeatureScaling(X)
	print X[14],Y[14]
	theta				= 	np.zeros((7,1)) 	# initailzing theta value
	learning_rate			= 	0.1
	J				=	CostFuction(X,Y,m,theta)
	print " Testing Initial Value of Cost Function %f"%J
	[theta,j_history]		=	Gradiant_descent(X,Y,theta,m,learning_rate,iterations)
	
	# trian cost function
	J_train  			=  	CostFuction(X,Y,m,theta)

	print " Testing Train Value of Cost Function %f"%J_train
	#test Gradiant desent

	test_data      			=	pd.read_csv("test.csv").as_matrix()
	x_test 				=	test_data[:,1:7]
	y_test 				=       test_data[:,0]
	m_test              		=  	len(y_test)
	x_test 				= 	np.c_[np.ones(m_test),x_test]
	x_test              		=   	FeatureScaling(x_test)
	j_test				=	CostFuction(x_test,y_test,m_test,theta)
	
	print " Testing Test Value of Cost Function %f"%j_test

	print theta
	
	# Predicting price for input dataset
	test_data1	=	[1,-0.38773777,0.24903439,-0.37208855,-0.21181928,1.11320682,-1.38888858]
	# Testing 
	
	test_data1	= 	np.asarray(test_data1)
	Predict1 	=	test_data1.dot(theta)
	print "Estimate Price of House is: $%f "%Predict1
	print "Original Price : $257500 "

	test_data2	=	[1,-0.3877495,-0.409448634,-1.0534574,-0.13330988,-0.84867380,0.256399555]
	test_data2	= 	np.asarray(test_data2)
	Predict2 	=	test_data2.dot(theta)
	print "Estimate Price of House is: $%f "%Predict2
	print "Original Price : $224000 "

	test_data3	=	[1,0.6824907,1.23663305,0.99609053,-0.25502121,1.11320682,0.88477959]
	test_data3	= 	np.asarray(test_data3)
	Predict3 	=	test_data3.dot(theta)
	print "Estimate Price of House is: $%f "%Predict3
	print "Original Price : $650000"

	#Predicting price based on input
	
	Output          =  InputData(theta)
	print "Estimate Price of House is: $%f"%Output



if __name__ == '__main__':
	LinearRegression()
