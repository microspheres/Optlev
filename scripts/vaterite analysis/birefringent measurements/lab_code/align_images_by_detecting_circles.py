import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join

import image_processing_functions as ipf
# image_list = ipf.return_cb_list()

# from skimage import data, color
from skimage.transform import hough_circle, rotate, rescale  # , warp, AffineTransform
from skimage.feature import peak_local_max, canny  # , corner_harris, corner_subpix, corner_peaks, plot_matches
# from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte, img_as_float
# from skimage.exposure import rescale_intensity
# from skimage.color import rgb2gray
# from skimage.measure import ransac
# from skimage.io import imread  # , imsave
from scipy.misc import imread, imsave

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
extension = "png"

scale_image = False
should_save = False

path_name = default_path + 'vaterite 40X 3-9-18\\'
file_list = glob(join(path_name, '*0b*.' + extension))

background_brightness, brightest_bg, c_b_names = ipf.brightness_lists(file_list)
print c_b_names

# c_b_names = ['a100b0', 'a80b20', 'a60b40', 'a40b60', 'a20b80',
#              'a0b100', 'a160b120', 'a140b140', 'a120b160', 'a100b180']
c_b_list = [path_name + c_b + '.' + extension for c_b in c_b_names]

hough_radii = [10]
start_j = [0, 420, 375, 260]
end_j = [70, 480, 410, 300]
start_k = [60, 20, 630, 520]
end_k = [100, 60, 665, 570]


def get_centers_list(list_of_images):
    m = len(list_of_images)  # should be 10
    n = len(start_j)  # should be 4
    list_of_sphere_centers = np.zeros((m, n, 2))
    for index, image in enumerate(list_of_images):
        # Load picture and detect edges
        edges = canny(image, sigma=3, low_threshold=10, high_threshold=50)
        for i, (j1, j2, k1, k2) in enumerate(zip(start_j, end_j, start_k, end_k)):
            # Detect two radii
            hough_res = hough_circle(edges[j1:j2, k1:k2], hough_radii)
            #         print "getting centers woo"
            centers = []
            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract two circles
                peaks = peak_local_max(h, num_peaks=1)
                #             print "    ", peaks
                peaks[0][0] += j1
                peaks[0][1] += k1
                centers.extend(peaks)
                #         print "putting centers into a list"
            # Put relevant centers in list_of_sphere_centers
            for c in centers:
                if list_of_sphere_centers[index][i][0] == 0. and list_of_sphere_centers[index][i][1] == 0.:
                    list_of_sphere_centers[index][i] = c
                else:
                    # print "duplicate center in", f[f.rfind('\\'):], "for", i
                    # print "                   ", list_of_sphere_centers[index][i]
                    # print "                   ", c
                    old_c = list_of_sphere_centers[index][i]
                    list_of_sphere_centers[index][i][0] = (c[0] + old_c[0]) // 2
                    list_of_sphere_centers[index][i][1] = (c[1] + old_c[1]) // 2
                    # print "    went into list_of_sphere_centers"
    return list_of_sphere_centers


def show_spheres_with_xs(list_of_images, list_of_sphere_centers):
    for f, spots in zip(list_of_images, list_of_sphere_centers):
        plt.figure()
        plt.imshow(imread(f, flatten=True))
        for s in spots:
            plt.plot(s[1], s[0], 'x', label=s)
        plt.legend()
        plt.show()


def show_all_edges(list_of_images):
    sample_image = list_of_images[0]
    plt.figure()
    plt.imshow(sample_image)
    plt.show(block=False)

    edges = canny(sample_image, sigma=3, low_threshold=10, high_threshold=50)
    for im in list_of_images[1:]:
        edges += canny(im, sigma=3, low_threshold=10, high_threshold=50)

    plt.figure()
    plt.imshow(edges)
    plt.show()


def get_information(curr_spots):
    curr_sphere_1 = curr_spots[0]
    curr_sphere_2 = curr_spots[-1]
    curr_angle = ipf.find_angle(curr_sphere_1, curr_sphere_2)
    curr_distance = ipf.find_distance(curr_sphere_1, curr_sphere_2)
    return curr_angle, curr_distance


def get_move_by(ref_spots, curr_spots):
    move_by = [0, 0]
    for rs, cs in zip(ref_spots, curr_spots):
        move_by[0] += rs[0] - cs[0]
        move_by[1] += rs[1] - cs[1]
    move_by = np.array(move_by) / len(ref_spots)
    return int(move_by[0]), int(move_by[1])


def show_sum(image_list):
    l, m, n = image_list.shape
    imsum = np.zeros((m, n))
    for im in image_list:
        print (im.dtype, np.median(im))
        imsum += im
    plt.imshow(imsum / l)
    plt.colorbar()
    plt.show()

    file_name = path_name + 'aligned\\imsum.png'
    if should_save:
        imsave(file_name, imsum.astype(np.uint8))


def save_aligned_image(imfile, norm_factor, angle, zoom_factor, move_parameters):
    """aligns the image and then saves a scaled version of it so there's less data to analyze"""
    im0 = imread(imfile, flatten=True)
    im1 = im0 * norm_factor
    if zoom_factor == 1.0:
        im3 = im1
    else:
        im2 = rescale(im1, zoom_factor)
        im3 = ipf.restore_original_size(im2, im0.shape)
    im4 = rotate(im3, angle)
    if move_parameters[0] == 0.:
        im5 = im4
    else:
        im5 = np.roll(im4, int(move_parameters[0]), axis=0)
    if move_parameters[1] == 0.:
        im6 = im5
    else:
        im6 = np.roll(im5, int(move_parameters[1]), axis=1)
    if scale_image:
        im6 = rescale(im6, 0.5)
    path = path_name + 'aligned'
    original_name = imfile[imfile.rfind('\\'):]
    file_name = path + original_name
    if should_save:
        # imsave(file_name, im6)  # .astype(np.uint8)
        # plt.imsave(file_name, im6, cmap=plt.get_cmap('gray'))
        imsave(file_name, im6.astype(np.uint8))
    return im6


def get_parameter_lists(c_b_list):
    ref_image = imread(c_b_list[0], flatten=True)
    m, n = ref_image.shape
    image_list = np.zeros((10, m, n))
    image_list[0] = ref_image
    for index, f in enumerate(c_b_list[1:]):
        image_list[index + 1] = imread(f, flatten=True)

    # show_all_edges(image_list)  # ############################################

    angle_list = np.zeros(10)
    zoom_list = np.zeros(10)
    move_list = np.zeros((10, 2))

    # centers_list[b_value][which_sphere][j_or_k]
    centers_list = get_centers_list(image_list)

    ref_angle, ref_distance = get_information(centers_list[0])

    rotated_and_zoomed_images = np.zeros((10, m, n))
    for index, curr_image in enumerate(image_list):
        curr_spots = centers_list[index]
        curr_angle, curr_distance = get_information(curr_spots)

        z = ref_distance / curr_distance
        zoom_list[index] = z

        new_image = rescale(curr_image, z)
        zoomed_image = ipf.restore_original_size(new_image, curr_image.shape)

        a = ref_angle - curr_angle
        angle_list[index] = a
        rotated_and_zoomed_images[index] = rotate(zoomed_image, a)

    new_centers_list = get_centers_list(rotated_and_zoomed_images)
    ref_spots = new_centers_list[0]

    translated_images = np.zeros((10, m, n))
    for index, curr_image in enumerate(rotated_and_zoomed_images):
        curr_spots = new_centers_list[index]
        x, y = get_move_by(ref_spots, curr_spots)
        move_list[index] = (x, y)
        print (x, y)
        tranim = np.roll(curr_image, x, axis=0)
        translated_images[index] = np.roll(tranim, y, axis=1)

    show_sum(translated_images)  # #####################################################
    # insert a dialog thing here so I can type in the limits
    return zoom_list, angle_list, move_list  # j_start_and_end, k_start_and_end


zoom_list, angle_list, move_list = get_parameter_lists(c_b_list)
image_stack = []  # pure rotated+zoomed+translated image
alpha_array = []  # in degrees
beta_array = []  # in degrees
print ('aligning and saving files')
for f in file_list:
    # for f in c_b_list:
    print (f)
    alpha, beta = ipf.read_name_degrees(f)
    # a, b = [t / 20 for t in [alpha, beta]]
    b = beta / 20
    angle = angle_list[b]
    zoom_factor = zoom_list[b]
    move_parameters = move_list[b]
    # norm_factor = brightest_bg[a] / background_brightness[a, b]
    norm_factor = 1.
    print (angle, zoom_factor, move_parameters, norm_factor)

    image_stack.append(save_aligned_image(f, norm_factor, angle, zoom_factor, move_parameters))
    alpha_array.append(alpha)
    beta_array.append(beta)
    print ("saved")

alpha = np.array(alpha_array) * np.pi / 180.
beta = - np.array(beta_array) * np.pi / 180.
minimum_intensity = np.min(background_brightness[background_brightness > 0])
background_brightness -= minimum_intensity
for i in range(len(image_stack)):
    image_stack[i] -= minimum_intensity

c_guess = float(background_brightness.max())
a, b = np.unravel_index(background_brightness.argmax(), background_brightness.shape)
a0_guess = a * 20. * np.pi / 180.
b0_guess = - b * 20. * np.pi / 180.
phi_guess, eta_guess, theta_guess = [0. for i in range(3)]
starting_parameters = [c_guess, phi_guess, eta_guess, theta_guess, a0_guess, b0_guess]

if should_save:
    np.savez(path_name + 'aligned\\image_stack.npz', image_stack=image_stack, alpha_radians=alpha, beta_radians=beta,
             starting_parameters=starting_parameters)  # add everything here
