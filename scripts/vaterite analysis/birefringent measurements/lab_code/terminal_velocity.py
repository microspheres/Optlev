# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import circle
from skimage.color import gray2rgb

micron_per_pixel = 410 / 1462.357  # micron / pixel

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
default_path = default_path + '2019-05_measurements\\'

folder_name_list = ['alumina_birefringence',
                    'corpuscular_silica_15um_birefringence',
                    'corpuscular_silica_15um_birefringence_2',
                    'gadi_vaterite_birefringence',
                    'gadi_vaterite_birefringence_2',
                    'german_8um_vaterite_birefringence',
                    'german_8um_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence',
                    'german_11um_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence_3',
                    'german_22um_silica_birefringence',
                    ]

sphere_labels = ['Alumina', '15um Corpuscular silica', 'House-made vaterite', '8um microParticles vaterite',
                 '11um microParticles vaterite', '22um microParticles silica']

save_images = True
show_all_plots = True
debugging = False  # only using the single sphere


def sinusoid(angle, phase, amplitude, constant):
    return amplitude * (1. + constant * np.cos(2. * angle + phase))


def get_circ_pts(folder_name, measure_locations, fit=False):
    # First we import all the images and their fits, as well as the locations of the spheres

    path_name = default_path + folder_name + "\\"

    npzfile_im_stack = np.load(path_name + 'aligned_image_stack.npz')
    image_stack = npzfile_im_stack['image_stack']

    n, nj, nk = image_stack.shape
    if debugging:
        print nj, nk

    npzfile_s_bg_loc = np.load(path_name + 'sphere_locations.npz')
    backgroundcoords = npzfile_s_bg_loc['background']

    if fit:
        background = [np.median(np.ravel(im)) for im in image_stack]
        min_bg = float(np.min(background))
        npzfile = np.load(path_name + 'aligned_simple_intensity_sinusoid_amplitudes.npz')
        constant = np.array(npzfile['constant'])
        amplitude = np.array(npzfile['amplitudes'])
        new_constant = amplitude * constant / (amplitude + min_bg)
        circularity_squared = 1 - new_constant ** 2
    else:
        brightest_pixels_im = np.array(np.max(image_stack, axis=0), dtype=float)
        darkest_pixels_im = np.array(np.min(image_stack, axis=0), dtype=float)
        constant_ar = (brightest_pixels_im - darkest_pixels_im) / (brightest_pixels_im + darkest_pixels_im)
        circularity_squared = 1 - constant_ar ** 2

    background_circularity_squared = np.median(circularity_squared[backgroundcoords])
    bg_circ = np.sqrt(np.abs(background_circularity_squared))

    delta_circ = []
    for radius, x, y in measure_locations:
        cj, ck = circle(y, x, radius)
        ck = ck[cj < nj]
        cj = cj[cj < nj]
        cj = cj[ck < nk]
        ck = ck[ck < nk]

        if debugging:
            image2 = gray2rgb(image_stack[0])
            image2[cj, ck] = (220, 20, 20)
            plt.figure()
            plt.imshow(image2)
            plt.ylabel(folder_name)
            plt.show()

        curr_circ_squared = np.median(circularity_squared[cj, ck])
        curr_circ = np.sqrt(curr_circ_squared)

        delta_circ.append(curr_circ)
    delta_circ = np.array(delta_circ)
    return delta_circ, bg_circ / (2. * np.sqrt(delta_circ))


# def get_circ_pts(folder_name, measure_locations, fit=False):
#     # First we import all the images and their fits, as well as the locations of the spheres
#
#     path_name = default_path + folder_name + "\\"
#
#     npzfile_im_stack = np.load(path_name + 'aligned_image_stack.npz')
#     image_stack = npzfile_im_stack['image_stack']
#
#     n, nj, nk = image_stack.shape
#     if debugging:
#         print nj, nk
#
#     npzfile_s_bg_loc = np.load(path_name + 'sphere_locations.npz')
#     backgroundcoords = npzfile_s_bg_loc['background']
#
#     if fit:
#         background = [np.median(np.ravel(im)) for im in image_stack]
#         min_bg = float(np.min(background))
#         npzfile = np.load(path_name + 'aligned_simple_intensity_sinusoid_amplitudes.npz')
#         constant = np.array(npzfile['constant'])
#         amplitude = np.array(npzfile['amplitudes'])
#         new_constant = amplitude * constant / (amplitude + min_bg)
#         circularity_squared = 1 - new_constant ** 2
#     else:
#         brightest_pixels_im = np.array(np.max(image_stack, axis=0), dtype=float)
#         darkest_pixels_im = np.array(np.min(image_stack, axis=0), dtype=float)
#         constant_ar = (brightest_pixels_im - darkest_pixels_im) / (brightest_pixels_im + darkest_pixels_im)
#         circularity_squared = 1 - constant_ar ** 2
#
#     background_circularity_squared = np.median(circularity_squared[backgroundcoords])
#     if background_circularity_squared > 0:
#         bg_circ = np.sqrt(background_circularity_squared)
#     else:
#         bg_circ = 0.
#
#     delta_circ = []
#     for radius, x, y in measure_locations:
#         cj, ck = circle(y, x, radius)
#         ck = ck[cj < nj]
#         cj = cj[cj < nj]
#         cj = cj[ck < nk]
#         ck = ck[ck < nk]
#
#         if debugging:
#             image2 = gray2rgb(image_stack[0])
#             image2[cj, ck] = (220, 20, 20)
#             plt.figure()
#             plt.imshow(image2)
#             plt.ylabel(folder_name)
#             plt.show()
#
#         curr_circ_squared = np.median(circularity_squared[cj, ck])
#         if curr_circ_squared > 0:
#             curr_circ = np.sqrt(curr_circ_squared)
#         else:
#             curr_circ = 0.
#
#         delta_circ.append(curr_circ - bg_circ)
#
#     return np.array(delta_circ)


# if debugging:
#     print get_circ_pts(folder_name_list[-2], measure_radii[-2])

# fitting factors that need to eventually be dealt with
# geometric_factor = 1
# sigma = 1  # anywhere between 2 and 10
geometric_factor = 0.1
sigma = 2  # anywhere between 2 and 10

# sphere_radii = np.array(sphere_diameter) * 1e-6 / 2. # meters
# sphere_volume = (4. / 3.) * np.pi * (sphere_radii) ** 3  # cubic meter

density_of_sphere = 2540  # kg/m^3 # compared to glass density of ~2520 kg/m^3
# sphere_mass = sphere_volume * density_of_sphere
boltzmann_constant = 1.38065e-23  # kg * m^2 / s^2 / K (joules per kelvin)
acceleration_due_to_gravity = 9.807  # m/s^2
wavelength_of_trapping_beam = 1064 * 1e-9  # m
# speed_of_light = 299792458.  # m/s
room_temperature = 295  # K
air_mass = 4.809e-26  # kg (kilograms)


def get_terminal_velocity(vacuum_pressure, delta_circ, delta_circ_error, sphere_diameter):
    # pressure in N/m^2 = 1e-2mbar
    # rest are unitless
    # velocity in Hz in the molecular flow regime (when mean free path is much larger than the sphere)

    terminal_velocity_num = acceleration_due_to_gravity * wavelength_of_trapping_beam * delta_circ * density_of_sphere
    terminal_velocity_dem = geometric_factor * sphere_diameter * sigma * vacuum_pressure
    terminal_velocity_sqr = np.sqrt(boltzmann_constant * room_temperature / (2 * np.pi * air_mass))

    terminal_velocity = terminal_velocity_num / terminal_velocity_dem * terminal_velocity_sqr / (2 * np.pi)
    terminal_velocity_error = np.abs(terminal_velocity * delta_circ_error / delta_circ)
    return terminal_velocity, terminal_velocity_error


# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

# now to actually make the scatter plot:
diameters_in_micron_list = [[], [], [], [], [], []]
# terminal_velocity_no_fit = [[], [], [], [], [], []]
terminal_velocity_fitted = [[], [], [], [], [], []]
error_list = [[], [], [], [], [], []]

marker_list = ['o', 's', '*', '^', 'v', 'd']
indexing = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5]

for folder, i in zip(folder_name_list, indexing):
    npzfile = np.load(default_path + folder + '\\measure_locations.npz')
    sphere_list = npzfile['spheres']
    measure_list = sphere_list  # npzfile['measure']

    radii_in_pixels, _, _ = zip(*sphere_list)
    diameters_in_micron = 2 * np.array(radii_in_pixels) * micron_per_pixel
    diameters_in_micron_list[i].extend(diameters_in_micron)

    # delta_circ_pts, error = get_circ_pts(folder, measure_list)
    # terminal_velocity_no_fit[i].extend(get_terminal_velocity(100, delta_circ_pts, diameters_in_micron * 1.e-6))

    delta_circ_pts, error = get_circ_pts(folder, measure_list, fit=True)
    terminal_velocity_points, error = get_terminal_velocity(4., delta_circ_pts, error, diameters_in_micron * 1.e-6)
    terminal_velocity_fitted[i].extend(terminal_velocity_points)
    error_list[i].extend(error)

# sphere_labels = ['Alumina', '15um Corpuscular silica', 'House-made vaterite', '8um microParticles vaterite',
#                  '11um microParticles vaterite', '22um microParticles silica']
sphere_labels = ['Sample ' + str(i + 1) for i in range(6)]

# plt.figure(figsize=(5, 4))
# for d, c, m, l in zip(diameters_in_micron_list, terminal_velocity_fitted, marker_list, sphere_labels):
#     plt.plot(d, c, m, fillstyle='none', label=l)
# plt.ylabel('Terminal frequency [Hz]')
# plt.xlabel('Diameter of microsphere [um]')
# plt.legend()
# if save_images:
#     plt.tight_layout()
#     plt.savefig(default_path + 'scatter_plot_diameter_velocity_fit_full.png')
# plt.show()
#
#
# plt.figure(figsize=(5, 4))
# for d, c, m, l in zip(diameters_in_micron_list, terminal_velocity_fitted, marker_list, sphere_labels):
#     plt.plot(d, c, m, fillstyle='none', label=l)
# plt.yscale('log')
# plt.ylabel('Terminal frequency [Hz]')
# plt.xlabel('Diameter of microsphere [um]')
# plt.legend()
# if save_images:
#     plt.tight_layout()
#     plt.savefig(default_path + 'scatter_plot_diameter_velocity_fit_logy_full.png')
# plt.show()
#
#
# plt.figure(figsize=(5, 4))
# for d, c, m, l in zip(diameters_in_micron_list, terminal_velocity_no_fit, marker_list, sphere_labels):
#     plt.plot(d, c, m, fillstyle='none', label=l)
# plt.ylabel('Terminal frequency [Hz]')
# plt.xlabel('Diameter of microsphere [um]')
# plt.legend()
# if save_images:
#     plt.tight_layout()
#     plt.savefig(default_path + 'scatter_plot_diameter_velocity_full.png')
# plt.show()


# plt.figure(figsize=(5, 4))
# plt.plot([2], [1000])
# plt.plot([4], [2000])
# for d, c, m, l in zip(diameters_in_micron_list, terminal_velocity_no_fit, marker_list, sphere_labels)[2:5]:
#     plt.plot(d, c, m, fillstyle='none', label=l)
# plt.yscale('log')
# plt.ylabel('Terminal frequency [Hz]')
# plt.xlabel('Diameter of microsphere [um]')
# plt.legend()
# if save_images:
#     plt.tight_layout()
#     plt.savefig(default_path + 'scatter_plot_diameter_velocity_logy_vaterite.png')
# plt.show()

plt.figure(figsize=(5, 4))
plt.plot([2], [30000])
plt.plot([4], [30000])
for d, c, m, e, l in zip(diameters_in_micron_list, terminal_velocity_fitted, marker_list, error_list, sphere_labels)[2:5]:
    plt.plot(d, c, m, fillstyle='none', label=l)
    # plt.errorbar(d, c, fmt=m, yerr=e, fillstyle='none', label=l)
plt.yscale('log')
plt.ylabel('Terminal frequency [Hz]')
plt.xlabel('Diameter of microsphere [$\mu$m]')
plt.legend()
plt.tight_layout()
# plt.savefig(default_path + 'errorbar_diameter_velocity_1e-2mbar.png')
plt.savefig(default_path + 'scatter_plot_diameter_velocity_4e-2mbar.png')
plt.show()
