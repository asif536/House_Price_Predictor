import matplotlib.pyplot as plt


# Function to plot the data
def plot_Data(x,y):
	plt.xlabel("Sqft_living")
	plt.ylabel("House_Price")
	plt.scatter(x,y,marker=',',c='g')
	plt.show()
