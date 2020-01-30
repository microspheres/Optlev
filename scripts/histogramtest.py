import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot
import glob
import scipy.optimize as opt

def gauss(x,a,b,c):
    g = c*np.exp(-0.5*((x-a)/b)**2)
    return g

def histo(c, bins, r):
    h,b = np.histogram(c, bins = bins, range=r)
    bc = np.diff(b)/2 + b[:-1]
    return [h, bc]

n_list = [15,25]

folder = r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\wenqiang_power_to_loose\reproducibility"

n_list = glob.glob(folder+"/*.txt")


fig=plt.figure()
rll = [(0,0.1), (0, 0.2)]
coll = ['k', 'r']
lss = ['-','--']
ddl=[15,25]
for rl, cl, l, dd, i in zip(rll,coll,lss,ddl,range(2)):
    data = n_list[i]

    a = np.loadtxt(data)

    b = np.transpose(a)

    h, bc = histo(b, 8, rl)

    popt, pcov = opt.curve_fit(gauss, bc, h)
    x = np.linspace(rl[0], rl[1], 100)

    print(popt)

    yerrs = np.sqrt(h)
    yerrs[yerrs==0] = 1

    if(i==0):
        plt.errorbar(bc, h, yerr = yerrs , fmt = "o", color=cl, label="$d = %d\ \mu$m"%(dd))
    else:
        plt.errorbar(bc, h, yerr = yerrs , fmt = "o", mfc='white', color=cl, label="$d = %d\ \mu$m"%(dd))
    plt.plot(x, gauss(x, *popt), color=cl, linestyle=l)

    #label=r"$p_{loss} = %.2f\pm%.2f$ mbar"%(popt[0],popt[1])
plt.ylim(0, 27)
plt.xlim(0, 0.2)
#plt.xscale('log')

plt.xlabel("Pressure [mbar]")
plt.ylabel("Counts")
plt.legend(loc='upper left', fontsize=10)
fig.set_size_inches(4,3)
plt.tight_layout()

plt.savefig("loss_press.pdf")

plt.show()






# print 'b' , b

# x=b
# print 'x',x

# plt.figure(1)
# plt.subplot(211+i)
# plt.hist(x,label='sphere size=' + str(num) +'um')

# i=i+1
# plt.xlabel('losing pressure(mbar)')
# plt.ylabel('count')
	
# 	plt.xlim(0, 0.200)

# 	if i==1:
# 		plt.title('histogram of losing pressure ')
# 	plt.legend()

# plt.savefig('losing power histogram')
# plt.show()
