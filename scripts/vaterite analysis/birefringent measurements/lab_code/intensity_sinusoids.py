from scipy.optimize import curve_fit
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

""" calculates sinusoid fit from turning only the top polarizer """
default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"

directory_name = '2019-05-09_reduced'

folder_name_list = ['alumina_birefringence',
                    'corpuscular_silica_15um_birefringence',
                    'corpuscular_silica_15um_birefringence_2',
                    'gadi_vaterite_birefringence',
                    'gadi_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence',
                    'german_22um_silica_birefringence',
                    'german_8um_vaterite_birefringence',
                    'german_8um_vaterite_birefringence_2'
                    ]
n_jobs = 16
using_linux = True
simple = True

# have a debugging option. It'll only look at one sphere's worth of space and you can plot stuff - it's faster.
debugging = False


def sinusoid(angle, phase, amplitude, constant):
    if simple:
        return amplitude * (1. + constant * np.cos(2. * angle + phase))
    else:
        return 0.5 * amplitude * (1. + np.sin(2. * constant) * np.cos(2. * angle + phase))


if simple:
    bounds = ([-np.pi, 0, -10], [np.pi, 1000, 10])
else:
    bounds = ([-np.pi, 0, -np.pi / 2], [np.pi, 1000, np.pi / 2])


def get_intensity_sinusoid_fit(folder_name):
    npzfile = np.load(folder_name + 'aligned_image_stack.npz')
    array_of_images = npzfile["image_stack"]

    # subtract off the darkest image when fitting the data points:
    background = npzfile['background']
    array_of_images = np.array(array_of_images) - np.min(background)
    array_of_images[array_of_images < 0] = 0

    if debugging:
        f = np.load(folder_name + 'single_sphere_coords.npz')
        start_j = f['start_j']
        end_j = f['end_j']
        start_k = f['start_k']
        end_k = f['end_k']
        array_of_images = array_of_images[:, start_j:end_j, start_k:end_k]

    top_angle_in_radians = npzfile["radians"]
    n, nj, nk = array_of_images.shape

    # # first get p0 for the fits
    #
    # brightest_pixels_im = np.max(array_of_images, axis=0)
    # darkest_pixels_im = np.min(array_of_images, axis=0)
    # con_p0 = 1 - darkest_pixels_im / brightest_pixels_im
    # amp_p0 = brightest_pixels_im / (1 + con_p0)
    # phi_p0 = np.zeros((nj, nk))
    # for j in range(nj):
    #     for k in range(nk):
    #         data = array_of_images[:, j, k]
    #         curr_ind = np.argmax(data)
    #         phi_p0[j, k] = -2 * top_angle_in_radians[curr_ind]

    def get_popt(data, curr_j, curr_k):

        # curr_p0 = (phi_p0[curr_j, curr_k], amp_p0[curr_j, curr_k], con_p0[curr_j, curr_k])

        data_and_radians = zip(data, top_angle_in_radians)
        curr_data_and_radians = [x for x in data_and_radians if x[0] < 255]
        # print np.array(curr_data_and_radians).shape
        curr_data, curr_radians = zip(*curr_data_and_radians)

        try:
            popt, _ = curve_fit(sinusoid, curr_radians, curr_data, bounds=bounds)  # , p0=curr_p0)
            return popt, curr_j, curr_k
        except:
            if debugging:
                plt.figure()
                plt.plot(curr_radians, curr_data)
                plt.xlabel('angle of top polarizer (radians)')
                plt.ylabel('intensity (arb)')
                plt.title('Failed pixel for fitting: (j, k) = (' + str(j) + ', ' + str(k) + ')')
                plt.show()
            print '        ' + str(curr_j) + '/' + str(nj) + ", " + str(curr_k) + '/' + str(nk) + ' failed'
            return (0., 0., 0.), curr_j, curr_k

    # get a grid for each value of alpha
    amplitudes = np.zeros([nj, nk])
    constants = np.zeros([nj, nk])
    phases = np.zeros([nj, nk])

    for k in range(10, nk - 10):  # taking off the top and bottom ten because of potential alignment fails
        popts = Parallel(n_jobs=n_jobs)(delayed(get_popt)(array_of_images[:, j, k], j, k) for j in range(10, nj - 10))
        for popt, j, _ in popts:
            amplitudes[j, k] = popt[-2]
            constants[j, k] = popt[-1]
            phases[j, k] = popt[-3]
        print "    done with column " + str(k) + " out of " + str(nk) + " columns"

    if simple:
        npz_name = folder_name + 'aligned_simple_intensity_sinusoid_amplitudes.npz'
    else:
        npz_name = folder_name + 'aligned_intensity_sinusoid_amplitudes.npz'
    np.savez(npz_name, amplitudes=amplitudes, phase=phases, constant=constants)


for folder in folder_name_list:
    print 'working on ' + folder
    if using_linux:
        path_name = directory_name + '/' + folder + '/'
    else:
        path_name = default_path + directory_name + '\\' + folder + '\\'
    get_intensity_sinusoid_fit(path_name)
