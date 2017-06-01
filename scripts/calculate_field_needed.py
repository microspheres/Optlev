import glob
import numpy as np
from VoltagevsAmplitude import getACAmplitudeGraphs
from dipole_fit_scale import get_param
from scipy.stats import linregress

path = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points/60.0_74.9_75.4'
max_limit = 1e-19 # N; we want the amplitude of the force at the drive frequency to be below this number
background = 5e-17 # N
desired_ratio = background/max_limit # we want to be able to see the peak at 2f so we know what the amplitude of the peak at f is

Efield, data_f, data_2f, fit_f, fit_2f = get_param(path)

data_ratio = np.divide(data_2f, data_f)
fit_ratio = np.divide(fit_2f, fit_f)

""" this gives the ratios as a function of the electric field
    so now let's extrapolate that data to get the desired ratio

    here Efield = m ratio + b """

data_m, data_b, data_r, data_p, data_err = linregress(data_ratio, Efield)
fit_m, fit_b, fit_r, fit_p, fit_err = linregress(fit_ratio, Efield)

req_E_data = data_m*desired_ratio + data_b
req_E_fit = fit_m*desired_ratio + fit_b

print 'data: ' + str(req_E_data)
print 'fit:  ' + str(req_E_fit)


# Make the plots requested here
file_list = glob.glob(path+"/*.h5")
getACAmplitudeGraphs(file_list, make_plots = True, zeroDC = True)
