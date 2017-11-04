import numpy as np
import matplotlib.pyplot as plt

with open('accuracy_big') as f:
	x = [i*0.1 for i in range(0, 11)]
	y = [float(i) for i in f]
	
	plt.plot(x,y)
	plt.xlabel("mu")
	plt.ylabel("Accuracy")
	plt.title("Prediction Accuracies (concated)")
	plt.show()
