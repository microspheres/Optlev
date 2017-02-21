import numpy as np
import scipy.stats as sp
import scipy.signal as ss
import matplotlib.pyplot as plt



half_length = 8192
max_val = 2**12-1

xvals = np.arange(-half_length/2,half_length/2)
yvals = np.arange(-half_length/2,half_length/2)

xtot = np.hstack( (xvals, xvals[::-1]) )
ytot = np.hstack( (yvals, yvals[::-1]) )

dtot = np.transpose( np.vstack( (xtot, ytot) ) )

np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\triangle_buffer.txt", dtot, delimiter=",",fmt="%d")

## square
xvals = np.arange(-half_length/4,half_length/4)
yvals = np.arange(-half_length/4,half_length/4)

xtot = np.hstack( (xvals, xvals[-1]*np.ones(half_length/2),
                   xvals[::-1], xvals[0]*np.ones(half_length/2)) )
ytot = np.hstack( (yvals[0]*np.ones(half_length/2),yvals,
                   yvals[-1]*np.ones(half_length/2),yvals[::-1]) )

dtot = np.transpose( np.vstack( (xtot*0.2, ytot) ) )
dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\square_buffer.txt", dtot, delimiter=",",fmt="%d")

## circle
t = np.linspace(0,2*np.pi,half_length*2)
xtot = np.round(half_length*np.sin(t))
ytot = np.round(half_length*np.cos(t))

dtot = np.transpose( np.vstack( (xtot, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\circle_buffer.txt", dtot, delimiter=",",fmt="%d")

## lissajous
t = np.linspace(0,2*np.pi,half_length*2)
xtot = np.round(half_length*np.sin(1*t))
ytot = np.round(half_length*np.cos(10*t))

dtot = np.transpose( np.vstack( (xtot*0.2, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\lissajous_buffer.txt", dtot, delimiter=",",fmt="%d")

## sinx
t = np.linspace(0,2*np.pi,half_length*2)
xtot = np.round(half_length*np.sin(10.*t))
ytot = np.round(0.*half_length*np.cos(10*t))

dtot = np.transpose( np.vstack( (xtot*0.2, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\sinx_buffer.txt", dtot, delimiter=",",fmt="%d")

## siny
t = np.linspace(0,2*np.pi,half_length*2)
xtot = np.round(0.*half_length*np.sin(1*t))
ytot = np.round(half_length*np.cos(10*t))

dtot = np.transpose( np.vstack( (xtot*0.2, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\siny_buffer.txt", dtot, delimiter=",",fmt="%d")

## two_traps
t = np.linspace(0,2*np.pi*half_length/2.,num = half_length*2)
xtot = np.abs(np.round(half_length*np.sin(1*t)))
ytot = np.round(half_length*0.*t)

dtot = np.transpose( np.vstack( (xtot, ytot) ) )

dtot = 1.0*max_val*dtot/np.max(dtot)
np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\two_traps.txt", dtot, delimiter=",",fmt="%d")

## gauss_dist

mod_z = True

n = 9

t = np.linspace(np.pi/2., 2.*np.pi + np.pi/2., half_length/2**n + 1.)
#t = np.linspace(0, 2.*np.pi, half_length/2**n + 1.)
triangle = ss.sawtooth(t, width = 0.5)[:-1]

triangletot = np.array([])

for i in range(2**n):
    triangletot = np.hstack([triangletot, triangle])

xint = np.arange(-2, 2, 0.000005)
cdf = 2.*(sp.norm.cdf(xint)-0.5)

s = np.interp(triangletot, cdf, xint, left = np.min(xint), right = np.max(xint))

t2 = np.linspace(0, 2.*np.pi, 2.*half_length + 1)
modx = np.cos(t2 - np.pi) + 1.
mody = np.cos(t2 - np.pi) + 1.

xtot = modx[: -1]*np.hstack([s, s])
ytot = mody[: -1]*np.hstack([s, s])

dtot = np.transpose( np.vstack( (xtot, ytot, np.zeros(len(ytot))) ) )

## just use a gaussian for z
dtot[:,2] = sp.norm.pdf(np.linspace(-3,3,len(xtot)))
dtot[:,2] -= dtot[0,2]

for d in range(3):
    dtot[:,d]= 1.0*max_val*dtot[:,d]/np.max(dtot[:,d])

emp_vec = np.load("z_template.npy")
emptot = np.interp( np.linspace(0,1,2*half_length),np.linspace(0,1,len(emp_vec)), emp_vec)*max_val
if not mod_z:
    emptot = np.interp( np.linspace(0,1,2*half_length),np.linspace(0,1,len(emp_vec)), emp_vec)*0.

## set x mod to zero for now
#dtot[:,0] = 0

#dtot[:,2] = emptot[::-1]
#dtot[:,2] = np.roll(dtot[:,2], 0)

plt.plot(dtot)
plt.plot(emptot)
plt.show()

np.savetxt(r"C:\GitHub\opt_lev\labview\fpga\gauss_cdf.txt", dtot, delimiter=",",fmt="%d")
