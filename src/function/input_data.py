import numpy as np

def InputData(theta):
	li=[1]
	print "Enter the No of Bedrooms,No of Bathrooms,Sqft_living,Sqft_lot,No of floors\n"
	for i in range(1,6):
		accept=input()
		li.append(accept)
		x=np.asarray(li)
	predict=np.dot(x,theta)
	return predict
	
	
	