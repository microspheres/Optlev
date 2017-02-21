import numpy as np

#################################################
fname = r"..\DAQ_settings\electrode_sweep.txt"


## dc offsets to sweep over
dc_list = np.linspace(-0.25, 0.05, 50 ) ## V

## list of electrode frequencies
#freq_list = np.array([0, 17, 19, 23, 29, 31, 37, 41]) ## Hz
freq_list = np.array([41, 0, 0, 0, 0, 0, 0, 0]) ## Hz

## list of drive amplitudes
drive_amp = 9. ## V
amp_list = drive_amp*np.ones_like(freq_list)
##################################################

par_list = []
for dc in dc_list:
    electrodes_to_use = 1.0*(freq_list > 0.)
    dc_list = dc * electrodes_to_use if np.abs(dc) > 0 else 0.
    par_list.append( np.hstack( [electrodes_to_use, amp_list, freq_list, dc_list] ) )

par_list = np.array(par_list)

np.savetxt(fname, par_list, delimiter=",", fmt="%.2f")
