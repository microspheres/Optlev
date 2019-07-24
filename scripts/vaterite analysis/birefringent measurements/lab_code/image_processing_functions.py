import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import ndimage

# from old_code import FindSpheres as fs

debugging = False


def read_name_degrees(file_name):
    """reads off angle(s) from file name
       change to add beta once applicable"""
    i = file_name.rfind('a') + 1
    j = file_name.find('.')
    k = file_name.find('b', i)
    alpha_in_degrees = int(file_name[i:k])
    beta_in_degrees = int(file_name[k + 1:j])
    return alpha_in_degrees, beta_in_degrees


def show_image(f, title='', block=True):
    # .astype(np.uint8)
    if type(f) is str:
        f = misc.imread(f, flatten=True)
    plt.figure()
    plt.imshow(f)
    plt.title(title)
    plt.colorbar()
    plt.show(block=block)


def find_angle(spot1, spot2):
    angle = math.degrees(math.atan((abs(spot1[1] - spot2[1])) / (abs(spot1[0] - spot2[0]))))
    return angle


def find_average_angle_to_rotate(spots, ref_spots):
    angles = []
    for i, (s1, r1) in enumerate(zip(spots[:-1], ref_spots[:-1])):
        for j, (s2, r2) in enumerate(zip(spots[i + 1:], ref_spots[i + 1:])):
            angles.append(find_angle(s1, s2) - find_angle(r1, r2))
    average_angle = np.average(angles)
    average_error = np.sqrt(np.average((np.array(angles) - average_angle) ** 2))
    return average_angle, average_error


def find_distance(spot1, spot2):
    return np.sqrt((spot1[0] - spot2[0]) ** 2 + (spot1[1] - spot2[1]) ** 2)


def find_zoom_amount(spots, ref_spots):
    zoom_amount = []
    for i, (s1, r1) in enumerate(zip(spots[:-1], ref_spots[:-1])):
        for j, (s2, r2) in enumerate(zip(spots[i + 1:], ref_spots[i + 1:])):
            zoom_amount.append(find_distance(r1, r2) / find_distance(s1, s2))
    average_zoom_amount = np.average(zoom_amount)
    average_error = np.sqrt(np.average((np.array(zoom_amount) - average_zoom_amount) ** 2))
    return average_zoom_amount, average_error


#

# def return_cb_list():
#     background_contrast = np.zeros([10, 10])
#     for i in fs.image_list:
#         if debugging: print i
#         alpha, beta = read_name_degrees(i)
#         # f = fs.get_normalized_image(i)
#         f = misc.imread(i, flatten=True)
#         background_contrast[alpha / 20, beta / 20] = np.median(f)
#     c_b_indexes = [np.argmax(background_contrast[:, i]) for i in range(10)]  # brightest a for each b
#     if debugging: print c_b_indexes
#     c_b_names = ['a' + str(a * 20) + 'b' + str(b * 20) for b, a in enumerate(c_b_indexes)]
#     return c_b_names
#

def brightness_lists(image_list, num_steps=10):
    """ 
    returns the background values of all the images 
    as well as a list of the image with highest contrast for each b
    """
    # sample_file = image_list[0]
    # path = sample_file[:sample_file.rfind('\\')]+'\\'
    # extension = sample_file[sample_file.rfind('.')+1:]
    background_brightness = np.zeros([num_steps, num_steps])
    # background_contrast = np.zeros([10, 10])
    for i in image_list:
        if debugging:
            print (i)
        print i
        alpha, beta = read_name_degrees(i)
        a = alpha / 20
        b = beta / 20
        print a, b, alpha, beta
        if debugging:
            print (alpha, beta)
        o = misc.imread(i, flatten=True)
        # f = fs.get_normalized_image(i)
        background_brightness[alpha / 20, beta / 20] = np.median(o)
        # background_contrast[alpha / 20, beta / 20] = np.median(f)
    print num_steps
    brightest_bg = [float(np.max(background_brightness[i])) for i in range(num_steps)]  # brightest bg for each a
    # c_b_indexes = [np.argmax(background_contrast[:, i]) for i in range(10)]  # brightest a for each b
    c_b_indexes = [np.argmax(background_brightness[:, i]) for i in range(num_steps)]  # brightest a for each b
    if debugging:
        plt.imshow(background_brightness)
        plt.show()
        print c_b_indexes
    c_b_names = ['a' + str(a * 20) + 'b' + str(b * 20) for b, a in enumerate(c_b_indexes)]
    return background_brightness, brightest_bg, c_b_names


def translate_image(image, move_by):
    # print 'dimensions of original image are: ', image.shape
    # for i in (0, 1):
    #     delta = int(get_move_by[i])
    #     print 'delta is ', delta
    #     added_dimensions = np.array(image.shape)
    #     added_dimensions[i] = abs(delta)
    #     added_zeros = np.zeros(added_dimensions)
    #     if delta < 0:  # delete from the front
    #         image = np.delete(image, np.s_[:-delta], i)
    #         image = np.append(image, added_zeros, axis=i)
    #     elif delta > 0:  # delete from the back
    #         image = np.append(added_zeros, image, axis=i)
    #         image = np.delete(image, np.s_[-delta:], i)
    new_image = np.roll(image, int(move_by[0]), axis=0)
    new_image = np.roll(new_image, int(move_by[1]), axis=1)
    print 'image has been translated'
    # print 'new image dimensions are: ', image.shape
    return new_image


def restore_original_size(image, original_shape):
    """ by adding zeros on the edges or deleting the edge values """
    old_m, old_n = original_shape
    new_m, new_n = image.shape
    # print 'dimensions are currently ', image.shape
    delta_m_front = (old_m - new_m) / 2
    delta_m_back = (old_m - new_m) - delta_m_front
    delta_n_front = (old_n - new_n) / 2
    delta_n_back = (old_n - new_n) - delta_n_front
    if new_m > old_m:  # then delta_m is negative
        image = np.delete(image, np.s_[delta_m_front:], axis=0)
        image = np.delete(image, np.s_[:-delta_m_back], axis=0)
    else:
        added_zeros_m_front = np.zeros([delta_m_front, new_n])
        added_zeros_m_back = np.zeros([delta_m_back, new_n])
        image = np.append(added_zeros_m_front, image, axis=0)
        image = np.append(image, added_zeros_m_back, axis=0)
    if new_n > old_n:  # then delta_n is negative
        image = np.delete(image, np.s_[delta_n_front:], axis=1)
        image = np.delete(image, np.s_[:-delta_n_back], axis=1)
    else:
        added_zeros_n_front = np.zeros([old_m, delta_n_front])
        added_zeros_n_back = np.zeros([old_m, delta_n_back])
        image = np.append(added_zeros_n_front, image, axis=1)
        image = np.append(image, added_zeros_n_back, axis=1)
    # print 'dimensions are back to ', image.shape
    return image


def align_image(im1, angle, zoom_factor, move_by):
    print 'dimensions of im1 are ', im1.shape
    # first, rotate
    im2 = misc.imrotate(im1, angle)
    print 'dimensions of im2 are ', im2.shape
    # next, zoom
    im3 = ndimage.zoom(im2, zoom_factor)
    print 'dimensions of im3 are ', im3.shape
    if im1.shape != im3.shape:
        im3 = restore_original_size(im3, im1.shape)
    # finally, translate
    im4 = translate_image(im3, move_by)
    print 'dimensions of im4 are ', im4.shape
    return im4


#
# def align_parameters(image_name, ref_spot1, ref_angle, ref_distance, s, blur=False, debug=False):
#     im1 = fs.get_normalized_image(image_name)
#     sphere1 = fs.get_sphere_coords(im1, 1, blur=blur, debug=debug)
#     if debug: print 'got sphere1 for', image_name
#     sphere2 = fs.get_sphere_coords(im1, 2, blur=blur, debug=debug)
#     if debug: print 'got sphere2 for', image_name
#     # first, rotate
#     angle = ref_angle - find_angle(sphere1, sphere2)
#     im2 = misc.imrotate(im1, angle)
#     # next, zoom
#     zoom_factor = ref_distance / find_distance(sphere1, sphere2)
#     im3 = ndimage.zoom(im2, zoom_factor)
#     if s != im3.shape:
#         im3 = restore_original_size(im3, s)
#     print 'looking for sphere after rotation and zoom'
#     sphere1_new = fs.get_sphere_coords(im3, 1, blur=blur, debug=debug)
#     if debug: print 'got sphere1_new for', image_name
#     move_by = np.array(ref_spot1) - np.array(sphere1_new)
#     move_by = -np.array([move_by[1], move_by[0]])
#     return angle, zoom_factor, move_by


def save_aligned_image(imfile, path_name, norm_factor, angle, zoom_factor, move_by, extension_without_period):
    """aligns the image and then saves a scaled version of it so there's less data to analyze"""
    im0 = misc.imread(imfile, flatten=True)
    im1 = im0 * norm_factor
    image = align_image(im1, angle, zoom_factor, move_by)
    # maybe do something with the brightness image here (the brightest image for each a)
    scaled_image = ndimage.zoom(image, 0.5)
    path = path_name + 'aligned'
    original_name = imfile[imfile.rfind('\\'):imfile.rfind('.')]
    file_name = path + original_name + '.' + extension_without_period
    misc.imsave(file_name, scaled_image.astype(np.uint8))


def crop_parameters(move_by):
    cropped_edges = np.zeros(4)  # tells you how much to crop by
    max_m = max(move_by[:, 0])
    min_m = min(move_by[:, 0])
    max_n = max(move_by[:, 1])
    min_n = min(move_by[:, 1])
    if max_m > 0: cropped_edges[0] = max_m
    if min_m < 0: cropped_edges[1] = -min_m
    if max_n > 0: cropped_edges[2] = max_n
    if min_n < 0: cropped_edges[3] = -min_n
    return tuple(cropped_edges)


def save_translated_image(imfile, path_name, move_by, cropped_edges, extension_without_period):
    original_image = misc.imread(imfile, flatten=True)
    aligned_image = translate_image(original_image, move_by)
    # now crop the image so none of the black is showing
    max_m, min_m, max_n, min_n = cropped_edges
    aligned_image = aligned_image[min_m:-max_m, min_n:-max_n]
    print 'dimensions of aligned image are ', aligned_image.shape

    path = path_name + 'aligned'
    original_name = imfile[imfile.rfind('\\'):imfile.rfind('.')]
    file_name = path + original_name + '.' + extension_without_period
    misc.imsave(file_name, aligned_image.astype(np.uint8))

#
# def save_every_sphere1(image_list):
#     background_brightness, brightest_bg, c_b_list = brightness_lists(fs.image_list)
#     print 'Got the brightness lists'
#     print 'c_b_list is ', c_b_list
#     print 'brightest_bg is ', brightest_bg
#
#     # brightest_bg = brightest bg for each a
#     # c_b_indexes = brightest a for each b
#     prescom_list = np.zeros([10, 2])
#     for i in c_b_list:
#         print 'finding sphere for', i
#         alpha, beta = read_name_degrees(i)
#         prescom_list[beta / 20] = fs.find_sphere(i, fs.sphere_center_1, fs.original_box_size_1, fs.sphere_size_1, 20)
#     for i in fs.image_list:
#         print 'trying to save sphere1 from ', i
#         alpha, beta = read_name_degrees(i)
#         curr_bg = background_brightness[alpha / 20, beta / 20]
#         norm = brightest_bg[alpha / 20] / curr_bg
#         if norm < 1.: print "norm was less than 1 so YOU NEED TO FIX brightest_bg !!!"
#         prescom = prescom_list[beta / 20]
#         fs.save_sphere(i, prescom, norm)
#         print "saved sphere1 from ", i
