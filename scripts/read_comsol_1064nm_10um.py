import re
import numpy as np
import matplotlib.pyplot as plt

## sphere radius
r = 5.

## first parse the list of parameters
t = open(r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\ComsolCO2\temp_data_emissiv0_1_multpow_1064_10um_bound.txt", 'rt')
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
d = np.loadtxt( r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\ComsolCO2\temp_data_emissiv0_1_multpow_1064_10um_bound.txt", skiprows=9 )


xvals = d[:,0]*1e6 ## in um
sortvals = np.argsort(xvals)

xvals = xvals[sortvals]
d = d[sortvals,3:]


## plot center temperature versus pressure
fig1=plt.figure()
fig2=plt.figure()

xx = 50 # intensity to be multiplied by the powerfactor

for pow_col in [0.5, 1., 1.5]:

    gpts = colpars[:,1] == pow_col
    tvp = []

    for cpt in np.argwhere(gpts):

        cdat = d[:, cpt].flatten()

        tcent = np.interp( 0, xvals.T, cdat)
        tgrad = (np.interp( -r, xvals.T, cdat) - np.interp( r, xvals.T, cdat))
        tvp.append( [float(colpars[cpt][0][0])/100., tcent, tgrad] )

    tvp = np.array(tvp)
    
    plt.figure(fig1.number)
    plt.semilogx( tvp[:-1,0], tvp[:-1,1], label="%g $\mu$W/$\mu$m$^2$"%(xx*pow_col))

    plt.figure(fig2.number)
    plt.loglog( tvp[:-1,0], np.abs(tvp[:-1,2]), label="%g $\mu$W/$\mu$m$^2$"%(xx*pow_col))

plt.figure(fig1.number)
plt.legend()
plt.xlabel('Pressure [mbar]')
plt.ylabel('Temperature [K]')
plt.ylim([0, 1000])
plt.savefig('t_v_p.pdf')

plt.figure(fig2.number)
plt.legend()
plt.xlabel('Pressure [mbar]')
plt.ylabel('Temperature gradient [K]')
plt.ylim([0,5])
plt.savefig('delt_v_p.pdf')

plt.show()
