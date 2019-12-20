
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
import glob
from scipy.optimize import curve_fit


list = [10,15,25]
for i in list:
	num = i
	name = str(num) 
	a=np.loadtxt(name + '.txt')

	print i
	b= np.transpose(a)
	#print 'b' , b

	x=b[0]
	y=b[1]
	if i>3:
		y = y/25*1.4
	# for i in range(len(x)):
		# if x[i]<14:
			# x[i]=0
		# else:
			# x[i]=0.16*(x[i]-14)/25*1000

	plt.scatter(x,y,label=  str(num) + r"um")
	
	plt.xlabel('pressure(mbar)')
	plt.ylabel('laser power(mW/mm^2)')
	plt.title('losing power')
	ax = plt.gca()
	ax.set_xscale('log')
	plt.legend()
plt.savefig('all_figure_size')
plt.show()
