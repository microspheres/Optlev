# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle

# micron_per_pixel = 210 / 748.356  # micron / pixel
micron_per_pixel = 200. / 714.001  # micron / pixel

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
default_path = default_path + '2019-05_measurements\\'

folder_name_list = ['alumina_birefringence',
                    'corpuscular_silica_15um_birefringence',
                    'corpuscular_silica_15um_birefringence_2',
                    'gadi_vaterite_birefringence',
                    'gadi_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence',
                    'german_11um_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence_3',
                    'german_22um_silica_birefringence',
                    'german_8um_vaterite_birefringence',
                    'german_8um_vaterite_birefringence_2'
                    ]

save_images = True
debugging = False  # only using the single sphere


def sinusoid(angle, phase, amplitude, constant):
    return amplitude * (1. + constant * np.cos(2. * angle + phase))


def save_fit_images(folder_name):
    # First we import all the images and their fits, as well as the locations of the spheres

    path_name = default_path + folder_name + "\\"

    npzfile_im_stack = np.load(path_name + 'aligned_image_stack.npz')
    image_stack = npzfile_im_stack['image_stack']
    image_stack = np.array(image_stack)
    radians = npzfile_im_stack['radians']

    npzfile = np.load(path_name + 'single_sphere_coords.npz')
    start_j = npzfile['start_j']
    end_j = npzfile['end_j']
    start_k = npzfile['start_k']
    end_k = npzfile['end_k']

    npzfile_fit_data = np.load(path_name + 'aligned_simple_intensity_sinusoid_amplitudes.npz')

    amplitudear = np.array(npzfile_fit_data['amplitudes'])
    constant_ar = np.array(npzfile_fit_data['constant'])

    background = [np.median(np.ravel(im)) for im in image_stack]
    constant_ar = amplitudear * constant_ar / (amplitudear + float(np.min(background)))

    phase_array = npzfile_fit_data['phase']
    if debugging:
        image_stack = image_stack[:, start_j:end_j, start_k:end_k]

    nj, nk = constant_ar.shape
    if debugging: print nj, nk

    def show_fit_parameter(parameter_grid, title, file_name, vmax=1):
        plt.figure()
        plt.title(title)
        plt.xlabel('distance (um)')
        plt.ylabel('distance (um)')
        plt.imshow(parameter_grid[10:-10, 10:-10],
                   extent=[0, (nk - 20) * micron_per_pixel, (nj - 20) * micron_per_pixel, 0], vmin=0, vmax=vmax)
        plt.colorbar()
        plt.tight_layout()
        if save_images:
            plt.savefig(path_name + file_name + '.png')
        plt.show()

    # For example, to plot the constant on the sinusoid, we type in:
    show_fit_parameter(np.abs(constant_ar), 'coefficient of sinusoid in fit', 'sinusoid_coefficient')

    # default_path + f + '\\measure_locations', spheres = s, measure = m
    npzfile = np.load(path_name + 'measure_locations.npz')
    measure_locations = npzfile['measure']
    sphere_locations = np.zeros((nj, nk))
    for radius, x, y in measure_locations:
        cj, ck = circle(y, x, radius)
        ck = ck[cj < nj]
        cj = cj[cj < nj]
        cj = cj[ck < nk]
        ck = ck[ck < nk]
        sphere_locations[cj, ck] = 1
    sphere_locations = np.array(sphere_locations, dtype=bool)

    npzfile_s_bg_loc = np.load(path_name + 'sphere_locations.npz')
    # sphere_locations = npzfile_s_bg_loc['spheres']
    backgroundcoords = npzfile_s_bg_loc['background']
    if debugging:
        sphere_locations = sphere_locations[start_j:end_j, start_k:end_k]
        backgroundcoords = backgroundcoords[start_j:end_j, start_k:end_k]

    # calculating circularity squared
    if debugging: print 'Now calculating the square of the circularity'

    circularity_squared = 1 - constant_ar ** 2

    sphere_circularity_squared = circularity_squared[sphere_locations]
    sphere_circularity_squared = sphere_circularity_squared[sphere_circularity_squared < 1]
    background_circularity_squared = circularity_squared[backgroundcoords]
    background_circularity_squared = background_circularity_squared[background_circularity_squared < 1]

    mean_sphere = np.average(sphere_circularity_squared)
    mean_bg = np.average(background_circularity_squared)

    sigma_sphere = np.sqrt(np.average((sphere_circularity_squared - mean_sphere) ** 2))
    sigma_bg = np.sqrt(np.average((background_circularity_squared - mean_bg) ** 2))

    circ_square_title = 'Circularity squared ($1 - c^2$)'
    mean_sphere_title = 'mean square sphere circularity ' + '%.3f' % mean_sphere + '$\\pm$' + '%.3f' % sigma_sphere
    mean_backgr_title = 'mean square background circularity ' + '%.3f' % mean_bg + '$\\pm$' + '%.3f' % sigma_bg
    circularity_squared_title = circ_square_title + '\nwith ' + mean_sphere_title + '\nand ' + mean_backgr_title
    show_fit_parameter(circularity_squared, circularity_squared_title, 'circularity_squared')

    # take the square root to get the circularity
    new_circ = circularity_squared.copy()
    new_circ[new_circ < 0] = 0
    new_circ = np.sqrt(new_circ)
    show_fit_parameter(new_circ, 'Circularity ($\\sqrt{1 - c^2}$) \nwith unphysical values set to 0', 'circularity')

    # now it's time to calculate the residuals

    try:
        npzfile = np.load(path_name + 'residuals.npz')
        residual_ar = npzfile['residuals']
    except:
        imfit = np.zeros(image_stack.shape)
        residual_ar = np.zeros((nj, nk))

        for j in range(10, nj - 10):
            for k in range(10, nk - 10):
                data = image_stack[:, j, k]
                a = amplitudear[j, k]
                p = phase_array[j, k]
                c = constant_ar[j, k]

                fit = sinusoid(radians, p, a, c)
                imfit[:, j, k] = fit

                # Now going to calculate error as chi-squared
                difference = data - fit
                residual_ar[j, k] = np.sum(difference ** 2) / len(radians)

        np.savez(path_name + 'residuals.npz', residuals=residual_ar, fit=imfit)

    reduced_residual_image = residual_ar[10:-10, 10:-10]
    reduced_residual_array = np.ravel(reduced_residual_image)
    mean_res = np.average(np.ravel(reduced_residual_image))
    sigma_res = np.sqrt(np.average((reduced_residual_array - mean_res) ** 2))
    res_title = 'mean $\\chi^2$ = ' + '%.3f' % mean_res + '$\\pm$' + '%.3f' % sigma_res
    show_fit_parameter(residual_ar, 'Residuals ($\\chi^2 (= \\sum{(data - fit)^2 / n})$)\nwith ' + res_title,
                       'residuals', vmax=50)

    def plot_fit(j, k):
        data = image_stack[:, j, k]
        a = amplitudear[j, k]
        p = phase_array[j, k]
        c = constant_ar[j, k]

        sorted_radians = np.linspace(np.min(radians), np.max(radians))
        fit = sinusoid(sorted_radians, p, a, c)

        plt.figure()
        plt.plot(radians, data, 'o')
        plt.plot(sorted_radians, fit)
        plt.xlabel('Angle of top polarizer (radians)')
        plt.ylabel('Intensity measured')
        plt.title('$\\chi^2=$' + str(residual_ar[j, k]) + ' for point $(j, k)$ = (' + str(j) + ', ' + str(k) + ')')
        plt.tight_layout()
        if save_images:
            plt.savefig(path_name + 'fit_plot_j' + str(j) + '_k' + str(k) + '.png')
        plt.show()

    # pick out six of the worst pixels in a sphere

    if debugging:
        single_sphere_locations = sphere_locations
        temp_res = residual_ar
    else:
        single_sphere_locations = sphere_locations[start_j:end_j, start_k:end_k]
        temp_res = residual_ar[start_j:end_j, start_k:end_k]
    single_sphere_residuals = np.zeros(single_sphere_locations.shape)
    single_sphere_residuals[single_sphere_locations] = temp_res[single_sphere_locations]
    residual_list = sorted(set(temp_res[single_sphere_locations]), reverse=True)

    for i in range(6):
        j, k = np.unravel_index(np.abs(single_sphere_residuals - residual_list[i]).argmin(), temp_res.shape)
        if debugging:
            plot_fit(j, k)
        else:
            plot_fit(j + start_j, k + start_k)


for folder in folder_name_list:
    save_fit_images(folder)
    print folder + ' done'
