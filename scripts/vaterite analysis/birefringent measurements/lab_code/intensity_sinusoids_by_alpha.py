# from birefringence_grids import jones_calculation as jones_calculation_single
# from birefringence_grids import get_chi_squared, default_path  # , show_image
# from scipy import misc
# from glob import glob
# from os.path import join

# import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from joblib import Parallel, delayed

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
folder_name = default_path + "12-3-19_vaterite\\"
d = 13.

npzfile = np.load(folder_name + 'image_stack.npz')
grid_of_images = npzfile["grid_of_images"]
nj, nk = (grid_of_images[0][0]).shape
beta = np.arange(9) * np.pi / 180.


def sinusoid(angle, phase, amplitude, constant):
    return 0.5 * amplitude * (1. + np.sin(2. * constant) * np.cos(2 * angle + phase))


def get_popt(data, j, k):
    popt, _ = curve_fit(sinusoid, beta, data, bounds=(0, [2 * np.pi, 200, 2 * np.pi]))
    return popt, j, k


amplitudes = np.zeros([9, nj, nk])
constants = np.zeros([9, nj, nk])
phases = np.zeros([9, nj, nk])
residuals = np.zeros([9, 9, nj, nk])
for a, images_by_b in enumerate(grid_of_images):
    for k in range(nk):
        popts = Parallel(n_jobs=1)(delayed(get_popt)(images_by_b[:, j, k], j, k) for j in range(nj))
        for popt, j, k in popts:
            data = images_by_b[:, j, k]  # 9 data points per pixel
            fit = sinusoid(beta, *popt)
            amplitudes[a, j, k] = popt[-2]
            constants[a, j, k] = popt[-1]
            phases[a, j, k] = popt[-3]
            residuals[a, :, j, k] = np.abs(data - fit) / np.maximum(data, fit)
        print 'done with column ' + str(k) + ' for alpha = ' + str(a * 20) + ' degrees'
    print 'done with alpha = ' + str(a * 20) + ' degrees'

np.savez(folder_name + 'intensity_sinusoid_by_alpha.npz', amplitudes=amplitudes, phase=phases, constant=constants,
         residuals=residuals)
