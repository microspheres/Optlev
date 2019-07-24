import numpy as np

""" force due to the trap laser on the vertical direction """
""" The useful function is: 
    integrated_power_on_sphere(sphere_radius_in_micron)   """

make_plot = False

waist = 25.0 * 1e-6  # waist at the sphere is 25 um

f = 25.0 * 1e-3  # 25.0 mm

wavelength = 1064. * 1e-9  # wavelength = 1064 nm

w0 = 0.772 * 1e-3  # with error 0.017*1e-3


def waist_focus_calculator(w0, light_wavelength, focal_distance_mm):
    """
    :param w0:
    :param light_wavelength:
    :param focal_distance_mm:
    :return:
    """
    aux = (focal_distance_mm * light_wavelength / np.pi) ** 2
    aux2 = 0.5 * (- np.sqrt(w0 ** 4 - 4. * aux) + 1. * w0 ** 2)
    return np.sqrt(aux2)


waist_focus = waist_focus_calculator(w0, wavelength, f)


def sphere_height(waist, waist_focus):
    """
    :param waist:
    :param waist_focus:
    :return:
    """
    return ((np.pi * waist_focus ** 2) / wavelength) * np.sqrt((waist / waist_focus) ** 2 - 1)


height = sphere_height(waist, waist_focus)
# print "height", height
# print "waits at focus", waist_focus
# print "NA", l / (pi * waist_focus)
# print "NA another way to calculate", w0 / f
# print "NA laser", l / (np.pi * w0)
# print "laser_waist", w0

n = 1.45  # index of refraction is 1.45 at 1064 nm

c = 299792458.0  # speed of light in m/s

power_of_laser = 1.  # watt


def power_on_sphere(beta, w0, f, laser_power):
    """
    :param beta: angle of incident ray
    :param w0:
    :param f: focal distance of the lens or the numerical aperature
    :param laser_power: obvious
    :return:
    """
    return (2.0 * laser_power / (np.pi * w0 ** 2)) * np.exp(-2 * (f * beta / w0) ** 2)


def integrated_power_on_sphere(sphere_diameter_in_micron, laser_power):
    """
    probably (hopefully) returns the power that the sphere feels
    :param sphere_diameter_in_micron: obvious
    :return:
    """
    sphere_radius = sphere_diameter_in_micron * 1e-6 / 2. # m
    d_beta = 0.00001
    # print sphere_radius / height
    # print w0 / f
    beta_max = np.min([sphere_radius / height, w0 / f])
    beta = np.arange(0, beta_max, d_beta)

    total_power = 0.
    for b in beta:
        curr_power = power_on_sphere(b, w0, f, laser_power)
        # print curr_power
        total_power =+ curr_power * b * d_beta

    return total_power



# # testing something
# sphere_diameters = [12.10798295, 11.91220638, 14.16211405, 14.11638275, 13.77495977, 12.4479054] # micron, from imageJ
# sphere_diameters = np.array(sphere_diameters)
#
# integrated_power_on_sphere(sphere_diameters)