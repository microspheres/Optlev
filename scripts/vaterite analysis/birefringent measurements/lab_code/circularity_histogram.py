from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

micron_per_pixel = 210 / 748.356  # micron / pixel

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
default_path = default_path + '2019-05-09\\'

folder_name_list = ['alumina_birefringence',
                    'corpuscular_silica_15um_birefringence',
                    'corpuscular_silica_15um_birefringence_2',
                    'gadi_vaterite_birefringence',
                    'gadi_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence',
                    'german_22um_silica_birefringence',
                    'german_8um_vaterite_birefringence',
                    'german_8um_vaterite_birefringence_2']


def gauss(x, mu, sigma, a):
    return a * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def save_circularity_histogram(folder_name):
    path_name = default_path + folder_name + "\\"

    npzfile_fit_data = np.load(path_name + 'aligned_simple_intensity_sinusoid_amplitudes.npz')

    # amplitudear = npzfile_fit_data['amplitudes']
    constant_ar = npzfile_fit_data['constant']
    # phase_array = npzfile_fit_data['phase']

    # nj, nk = constant_ar.shape

    circularity_squared = 1 - constant_ar ** 2

    npzfile_s_bg_loc = np.load(path_name + 'sphere_locations.npz')
    sphere_locations = npzfile_s_bg_loc['spheres']
    backgroundcoords = npzfile_s_bg_loc['background']

    sphere_circularity_squared = circularity_squared[sphere_locations]
    background_circularity_squared = circularity_squared[backgroundcoords]

    # mean_sphere = np.average(sphere_circularity_squared)
    # mean_background = np.average(background_circularity_squared)
    #
    # sigma_sphere = np.sqrt(np.average((sphere_circularity_squared - mean_sphere) ** 2))
    # sigma_background = np.sqrt(np.average((background_circularity_squared - mean_background) ** 2))
    #
    # print mean_sphere, sigma_sphere
    # print mean_background, sigma_background

    num_sp_bins = 20
    num_bg_bins = 20

    spy, spbins = np.histogram(sphere_circularity_squared, bins=20)
    bgy, bgbins = np.histogram(background_circularity_squared, bins=20)

    ratio_to_multiply_bg_bins = float(np.max(bgy)) / float(np.max(spy))

    if ratio_to_multiply_bg_bins > 1:
        num_bg_bins = int(num_bg_bins * ratio_to_multiply_bg_bins)
    else:
        num_sp_bins = int(num_sp_bins / ratio_to_multiply_bg_bins)

    # plt.figure()
    # plt.hist(sphere_circularity_squared, bins=num_sp_bins)
    # plt.hist(background_circularity_squared, bins=num_bg_bins)
    # plt.show()

    spy, spbins = np.histogram(sphere_circularity_squared, bins=num_sp_bins)
    bgy, bgbins = np.histogram(background_circularity_squared, bins=num_bg_bins)

    spx = spbins[1:]
    bgx = bgbins[1:]

    x = np.linspace(min(np.append(bgx, spx)), max(np.append(bgx, spx)), 100)

    sp_popt, pcov = curve_fit(gauss, spx, spy)
    # sp_mu2 = sp_popt[0]

    bg_popt, pcov = curve_fit(gauss, bgx, bgy)
    # bg_mu2 = bg_popt[0]
    #
    # if bg_mu2 > 0:
    #     bg_mu = np.sqrt(bg_mu2)
    # else:
    #     bg_mu = 0
    #
    # if sp_mu2 > 0:
    #     sp_mu = np.sqrt(sp_mu2)
    # else:
    #     sp_mu = 0

    plt.figure(figsize=(8, 5))

    plt.errorbar(spx, spy, yerr=np.sqrt(spy), fmt='.', label='sphere circularity histogram')
    plt.errorbar(bgx, bgy, yerr=np.sqrt(bgy), fmt='.', label='background circularity histogram')
    plt.plot(x, gauss(x, *sp_popt), label='sphere circularity gaussian fit')
    plt.plot(x, gauss(x, *bg_popt), label='background circularity gaussian fit')

    plt.xlabel('circularity squared values')
    plt.ylabel('number [arb]')
    plt.title('$\mu_b = $' + str(bg_popt[0]) + ' and $\mu_s = $' + str(sp_popt[0]))

    plt.legend()
    plt.savefig(path_name + 'histogram.png')
    plt.show()

    # print bg_mu, sp_mu


for folder in folder_name_list:
    save_circularity_histogram(folder)
