# from scipy.optimize import curve_fit
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
default_path = default_path + '2019-05-09\\'

folder_name_list = [
                    # 'alumina_birefringence',
                    # 'corpuscular_silica_15um_birefringence',
                    # 'corpuscular_silica_15um_birefringence_2',
                    # 'gadi_vaterite_birefringence',
                    # 'gadi_vaterite_birefringence_2',
                    # 'german_11um_vaterite_birefringence',
                    'german_11um_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence_3',
                    # 'german_22um_silica_birefringence',
                    # 'german_8um_vaterite_birefringence',
                    # 'german_8um_vaterite_birefringence_2'
                    ]

save_images = True


def sinusoid(angle, phase, amplitude, constant):
    return amplitude * (1. + constant * np.cos(2. * angle + phase))


def gauss(x, mu, sigma, a):
    return a * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)


def get_num_bins(data, starting_bins):
    desired_bins_per_sigma = 5
    y, bins = np.histogram(data, bins=starting_bins)
    mean = np.average(data)
    sigma = np.sqrt(np.average((data - mean) ** 2))
    num_bins_per_sigma = sigma / (bins[1] - bins[0])
    return mean, sigma, int(starting_bins * desired_bins_per_sigma / num_bins_per_sigma)


def make_single_histogram(constant_ar, sphere_locations, backgroundcoords, path_name, fitless=False):

    sphere_constant = np.abs(constant_ar[sphere_locations])
    background_constant = np.abs(constant_ar[backgroundcoords])

    # histogram time!

    mean_sphere, sigma_sphere, num_sp_bins = get_num_bins(sphere_constant, 20)
    mean_bg, sigma_bg, num_bg_bins = get_num_bins(background_constant, 100)

    spy, spbins = np.histogram(sphere_constant, bins=num_sp_bins)
    bgy, bgbins = np.histogram(background_constant, bins=num_bg_bins)

    spx = (spbins[1:] + spbins[:-1]) / 2.
    bgx = (bgbins[1:] + bgbins[:-1]) / 2.

    plt.figure(figsize=(8, 5))

    plt.errorbar(spx, spy / float(max(spy)), yerr=np.sqrt(spy) / float(max(spy)), fmt='o',
                 label='Sphere circularity histogram')
    plt.errorbar(bgx, bgy / float(max(bgy)), yerr=np.sqrt(bgy) / float(max(bgy)), fmt='^',
                 label='Background circularity histogram')

    plt.xlabel('Values of $\scrB$ in $\scrI=A(1 + B\\ \\cos(2\\theta + \\phi))$')
    plt.ylabel('Histogram [normalized]')

    bg_mu_title = '$\\mu_b=$%.3f' % mean_bg + '$\\pm$%.3f' % sigma_bg
    sp_mu_title = '$\\mu_s=$%.3f' % mean_sphere + '$\\pm$%.3f' % sigma_sphere
    plt.title(bg_mu_title + ' and ' + sp_mu_title)

    plt.legend()
    if save_images:
        if fitless:
            save_path = path_name + 'histogram_constant_fitless.png'
        else:
            save_path = path_name + 'histogram_constant.png'
        plt.savefig(save_path)
    plt.show()


def save_fitless_coefficient_histogram(folder_name):
    # First we import all the images and their fits, as well as the locations of the spheres

    path_name = default_path + folder_name + "\\"

    npzfile_im_stack = np.load(path_name + 'aligned_image_stack.npz')
    image_stack = npzfile_im_stack['image_stack']

    npzfile_s_bg_loc = np.load(path_name + 'sphere_locations.npz')
    # sphere_locations = npzfile_s_bg_loc['spheres']
    backgroundcoords = npzfile_s_bg_loc['background']

    threshold = imread(path_name + 'summed_image_edges.png', flatten=True)
    background = np.array(threshold, dtype=bool)
    sphere_locations = np.bitwise_xor(np.ones(threshold.shape, dtype=bool), background)

    brightest_pixels_im = np.array(np.max(image_stack, axis=0), dtype=float)
    darkest_pixels_im = np.array(np.min(image_stack, axis=0), dtype=float)
    constant_ar = (brightest_pixels_im - darkest_pixels_im) / (brightest_pixels_im + darkest_pixels_im)
    # amplitudear = (brightest_pixels_im + darkest_pixels_im) / 2.

    make_single_histogram(constant_ar, sphere_locations, backgroundcoords, path_name, fitless=True)


def save_both_coefficient_histograms(folder_name):
    # First we import all the images and their fits, as well as the locations of the spheres

    path_name = default_path + folder_name + "\\"

    npzfile_im_stack = np.load(path_name + 'aligned_image_stack.npz')
    image_stack = npzfile_im_stack['image_stack']

    npzfile_fit_data = np.load(path_name + 'aligned_simple_intensity_sinusoid_amplitudes.npz')

    constant_ar = npzfile_fit_data['constant']

    npzfile_s_bg_loc = np.load(path_name + 'sphere_locations.npz')
    # sphere_locations = npzfile_s_bg_loc['spheres']
    backgroundcoords = npzfile_s_bg_loc['background']

    threshold = imread(path_name + 'summed_image_edges.png', flatten=True)
    background = np.array(threshold, dtype=bool)
    sphere_locations = np.bitwise_xor(np.ones(threshold.shape, dtype=bool), background)

    make_single_histogram(constant_ar, sphere_locations, backgroundcoords, path_name)

    # Okay, now for the fitless calculation!!!

    brightest_pixels_im = np.array(np.max(image_stack, axis=0), dtype=float)
    darkest_pixels_im = np.array(np.min(image_stack, axis=0), dtype=float)
    constant_ar = (brightest_pixels_im - darkest_pixels_im) / (brightest_pixels_im + darkest_pixels_im)
    # amplitudear = (brightest_pixels_im + darkest_pixels_im) / 2.

    make_single_histogram(constant_ar, sphere_locations, backgroundcoords, path_name, fitless=True)


for folder in folder_name_list:
    save_both_coefficient_histograms(folder)
# save_both_coefficient_histograms(folder_name_list[-2])
