import numpy as np

def InputData():
	li=[1]
	print"Enter the No of Bedrooms,No of Bathrooms,Sqft_living,Sqft_lot,No of floors,zipcode(eg 98010)\n"
	for i in range(1,7):
		accept=float(input())
		li.append(accept)
		x=np.asarray(li)
	return x
	
	
	
