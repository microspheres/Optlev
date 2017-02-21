## create a csv file with the necessary voltage vs stage position to play with
## the DAQ

import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt

##############################################
output_file = r"C:\Data\20160310\bead1\chameleon_force_20160310.txt"
cant_pos_at_10V = 20. ## um
beta = 1e6 ## pick a reasonable beta to give observable force at 1V
##############################################

cant_V_per_um = 1./8

## first get the force vs position from Maxime's chameleon calculation
cant_pos = np.linspace(cant_pos_at_10V,cant_pos_at_10V+80.,1e3)*1e-6 ## cant pos, m
cham_force = bu.get_chameleon_force( cant_pos )*beta

## now convert the pos in m to V and write the file
cant_volt = 10. + (cant_pos_at_10V - cant_pos*1e6)*cant_V_per_um

## swap to increasing order and add a point at 10 V
cant_volt = np.hstack( (cant_volt[::-1], 10.01) )
cham_force = np.hstack( (cham_force[::-1], cham_force[0]) )

## now find the voltage that's needed to give the desired chameleon
## force at each position.  Here we just use a 1/r^2 fit to get
## roughly the right thing
bias_for_fit = 5.0 #volts
Afit = 1.2e-11  #coefficients for DC bias at 10V
def ffn(x):
    return Afit*(1./(x+50.))**2

## make force curve for 1V
elec_force = ffn( cant_pos*1e6 ) * (1./bias_for_fit)**2
elec_force = np.hstack( (elec_force[::-1], elec_force[0]) )

force_ratio = cham_force / elec_force
elec_dcvolt_vs_volt = np.sqrt(force_ratio)

outarr = np.vstack( (cant_volt, elec_dcvolt_vs_volt) ).T
np.savetxt( output_file, outarr, delimiter="," )

plt.figure()
plt.plot( cant_volt, elec_dcvolt_vs_volt )
plt.xlabel( "Cantilever set voltage [V]" )
plt.ylabel( "Electrode set voltage [V]" )

plt.show()
