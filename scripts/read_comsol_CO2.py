import re
import numpy as np
import matplotlib.pyplot as plt

## sphere radius
r = 7.5

## first parse the list of parameters
t = open(r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\ComsolCO2\temp_data_emissiv0_1_multpow_20200127_3.txt", 'rt')
l = t.readlines()
t.close()

pararr = re.split(r"T \(K\) \@ \d+\: ", l[8].strip())

colpars = []
for p in pararr[1:]:

    parts = p.split(',')
    print(parts)

    colpars.append( [float( parts[0][2:-3] ), float( parts[1][8:-1] )])

colpars = np.array(colpars)

## now get the data
d = np.loadtxt( r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\ComsolCO2\temp_data_emissiv0_1_multpow_20200127_3.txt", skiprows=9 )


xvals = d[:,0]*1e6 ## in um
sortvals = np.argsort(xvals)

xvals = xvals[sortvals]
d = d[sortvals,3:]


## plot center temperature versus pressure
fig1=plt.figure()
fig2=plt.figure()

xx = 250 # intensity to be multiplied by the powerfactor

Press = []
T = []
Power = []

for pow_col in [0.01, 0.03, 0.1, 0.3, 0.7, 1]:

    gpts = colpars[:,1] == pow_col
    tvp = []

    for cpt in np.argwhere(gpts):

        cdat = d[:, cpt].flatten()

        tcent = np.interp( 0, xvals.T, cdat)
        tgrad = (np.interp( -r, xvals.T, cdat) - np.interp( r, xvals.T, cdat))
        tvp.append( [float(colpars[cpt][0][0])/100., tcent, tgrad] )

    tvp = np.array(tvp)
    
    plt.figure(fig1.number)
    plt.semilogx( tvp[:-1,0], tvp[:-1,1], label="%g mW/mm$^2$"%(xx*pow_col))
    Press.append(tvp[:-1,0])
    T.append(list(tvp[:-1,1]))
    Power.append(xx*pow_col)
    

    plt.figure(fig2.number)
    plt.loglog( tvp[:-1,0], np.abs(tvp[:-1,2]), label="%g mW/mm$^2$"%(xx*pow_col))

plt.figure(fig1.number)
plt.legend()
plt.xlabel('Pressure [mbar]')
plt.ylabel('Temperature [K]')
plt.ylim([0, 1000])
plt.savefig('t_v_p.pdf')

plt.figure(fig2.number)
plt.legend()
plt.xlabel('Pressure [mbar]')
plt.ylabel('Temperature difference [K]')
plt.ylim([0,5])
plt.savefig('delt_v_p.pdf')


import scipy.special
import scipy.optimize as opt
from scipy import interpolate

from mpl_toolkits import mplot3d

Press = list(Press[0])
Intensity = Power

f = interpolate.interp2d(Press, Intensity, T, kind='linear')

Press_new = np.logspace(-3, 3, 100)
Intensity_new = np.linspace(250*0.01, 250, 100)
T_new = f(Press_new, Intensity_new)
print np.shape(T_new)
plt.figure()
plt.loglog(Press, T[0], 'ro-', Press_new, T_new[0], 'bo')


xnew, ynew = np.meshgrid(Press_new, Intensity_new)
x, y = np.meshgrid(Press, Intensity)

plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(np.log10(xnew), np.log10(ynew), np.log10(T_new), rstride=1, cstride=1,  cmap='viridis', edgecolor='none')
ax.scatter3D(np.log10(x), np.log10(y), np.log10(T), cmap='viridis', edgecolor='none')

plt.figure()
# intensity eq = 180mW/mm2
Ieq = 180
plt.loglog(Press_new, f(Press_new, Ieq), "ro-")
plt.loglog(Press_new, f(Press_new, Ieq/3), "bo-")

plt.figure()
intt = Press_new
for i in range(len(intt)):
    g = f(Press_new[i], i + 180)
    plt.loglog(Press_new[i], g, "bo-")





plt.show()

# def func(p, A, C):
#     C = np.abs(C)
#     f = A*(np.arctan( (1 - C*Press[0]) ) + np.pi/2) + 293.4
#     return f


# plt.figure()
# for i in range(len(Press)):

#     popt, pcov = opt.curve_fit(func, Press[i], T[i], maxfev = 10000)
#     plt.loglog(Press[i], T[i], "o")
#     plt.loglog(Press[i], func(Press[i], *popt))
