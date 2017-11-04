import numpy as np
import matplotlib.pyplot as plt

Dir = 'results/p2/small/'

for i in range(11):
	f = open(Dir + 'training_log_' + str(i * 0.1) + '.txt')
	inputs = f.readlines();
	x = [j for j in range(len(inputs))]
	y = [float(j) for j in inputs]
	
	f.close()
	
	plt.plot(x,y,label=str(i * 0.1))

plt.legend()
plt.title('Log likelihood in different iterations')
plt.xlabel('Interation')
plt.ylabel('Log Likelihood')

plt.show()
