import glob
import numpy as np
import matplotlib.pyplot as plt
from VoltagevsAmplitude import getACAmplitudeGraphs
from dipole_fit_scale import get_param, plot_amplitude_data_raw
from scipy.stats import linregress

path = '/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge43_whole_points/60.0_74.9_75.4'
background = 2e-17 # N

"""The following two parameters will be calculated later."""
#max_limit = f(E) # N; we want the amplitude of the force at the drive frequency to be below this number
#desired_ratio = background/max_limit # we want to be able to see the peak at 2f so we know what the amplitude of the peak at f is

Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f = get_param(path)

data_ratio = np.divide(data_2f, data_f) # unitless
fit_ratio = np.divide(fit_2f, fit_f) # unitless

# here I'm estimating the error bars
binF = 0.0762939453125 # got this from VoltagevsAmplitude.py
conversion = 3.7139625927e-13
data_error = 5e-5*conversion/binF # got 5e-5 by estimating by eye from plot_data_quick.py
data_ratio_err = np.multiply(np.sqrt(np.power(np.divide(data_error,data_f),2) + np.power(np.divide(data_error,data_2f),2)), data_ratio)
fit_ratio_err = np.multiply(np.sqrt(np.power(np.divide(err_f,fit_f),2) + np.power(np.divide(err_2f,fit_2f),2)), fit_ratio)

""" this gives the ratios as a function of the electric field
    so now let's extrapolate that data to get the desired ratio

    here Efield = m ratio + b 
    where m and b are in V/m """

data_m, data_b, data_r, data_p, data_err = linregress(data_ratio, Efield)
fit_m, fit_b, fit_r, fit_p, fit_err = linregress(fit_ratio, Efield)

""" Here's where it gets tricky, because max_limit is a function of the electric field: """
desired_measurement = 1e-21 # fractions of an electron charge
expected_measured_charge = 1.698e-4 * desired_measurement # C

# max_limit = expected_measured_charge * req_E # N
# desired_ratio = background / max_limit # unitless
# req_E = m * desired_ratio + b
#       = m * background / max_limit + b
#       = m * background / (expected_measured_charge * req_E) + b
#       = m * background / (expected_measured_charge * req_E) + b # V/m

# req_E**2 = m*background/expected_measured_charge + b*req_E
# 0 = req_E**2 - b*req_E - m*background/expected_measured_charge

# quadratic formula: x = (-b +/- sqrt(b^2 - 4*a*c))/(2*a)
four_a_c_data = 4*data_m*background/expected_measured_charge
four_a_c_fit = 4*fit_m*background/expected_measured_charge
two_a = 2
b_data = data_b
b_fit = fit_b
# req_E = (b_quad +/- np.sqrt(b_quad**2 + four_a_c))/two_a
# but the negative solutions don't actually make sense so we'll ignore them.

req_E_data = (b_data + np.sqrt(b_data**2 + four_a_c_data))/two_a
req_E_fit = (b_fit + np.sqrt(b_fit**2 + four_a_c_fit))/two_a # V/m

print ' '
print 'required electric field from data: ' + str(req_E_data) + ' +/- ' + str(data_err) + ' V/m'
print 'required electric field from fit:  ' + str(req_E_fit) + ' +/- ' + str(fit_err) + ' V/m'
print ' ===== OR ===== '
print 'required electric field from data: ' + str(req_E_data/1e6) + ' +/- ' + str(data_err/1e6) + ' kV/mm'
print 'required electric field from fit:  ' + str(req_E_fit/1e6) + ' +/- ' + str(fit_err/1e6) + ' kV/mm'
print ' '


# Make the plots requested here

file_list = glob.glob(path+"/*.h5")
getACAmplitudeGraphs(file_list, make_plots = True, zeroDC = True) # plot PSDs on top of each other

plot_amplitude_data_raw(path, Efield, data_f, data_2f, fit_f, fit_2f, err_f, err_2f) # plot the line and parabola

plt.figure()
plt.errorbar(Efield, data_ratio, yerr = data_ratio_err, label = "data")
plt.errorbar(Efield, fit_ratio, yerr = fit_ratio_err, label = "fit")
plt.legend()
plt.xlabel('Electric field [V/m]')
plt.ylabel('Ratio of force at 2f to force at f')
plt.title('Lines used in calculation')
plt.show()
