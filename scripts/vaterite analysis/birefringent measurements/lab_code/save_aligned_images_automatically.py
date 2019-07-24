import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny

directory_name = r"C:\Users\Sumita\Documents\Research\Microspheres\birefringent measurements\sphere images"
directory_name = directory_name + r'\2019-05-09'  # trying out getting rid of alignments that seem to fail

pixels_per_micron = 748.356 / 210.

folder_name_list = [# 'alumina_birefringence',
                    # 'corpuscular_silica_15um_birefringence',
                    # 'corpuscular_silica_15um_birefringence_2',
                    # 'gadi_vaterite_birefringence',
                    # 'gadi_vaterite_birefringence_2',
                    # 'german_11um_vaterite_birefringence',
                    # 'german_11um_vaterite_birefringence_2',
                    'german_11um_vaterite_birefringence_3',
                    # 'german_22um_silica_birefringence',
                    # 'german_8um_vaterite_birefringence',
                    # 'german_8um_vaterite_birefringence_2'
                    ]


def align_images_in_folder(folder_name):
    # First we load the image stack and its associated polarizer angle
    print 'loading image stack for'
    print folder_name
    npzfile = np.load(folder_name + '\\image_stack.npz')
    image_stack = npzfile['image_stack']
    background = npzfile['background']
    degrees = npzfile['degrees']

    n, nj, nk = image_stack.shape

    edges_stack = np.zeros((n, nj, nk))

    for i in range(n):
        edges_stack[i] = canny(image_stack[i], sigma=8)

    plt.figure(figsize=(20, 10))
    plt.imshow(np.sum(edges_stack, axis=0))
    plt.colorbar()
    # plt.title(folder_name + '\n sum of unaligned edges')
    plt.show()

    # Pick a sphere to use for alignment

    npzfile = np.load(folder_name + '\\single_sphere_coords.npz')
    start_j = npzfile['start_j']
    end_j = npzfile['end_j']
    start_k = npzfile['start_k']
    end_k = npzfile['end_k']

    cropped_edge_array = edges_stack[:, start_j:end_j, start_k:end_k]

    plt.figure(figsize=(10, 5))
    plt.imshow(np.sum(cropped_edge_array, axis=0))
    plt.colorbar()
    # plt.title(folder_name + '\n cropped sum of unaligned edges')
    plt.show()

    # # Align with np.roll and maximizing total brightness

    radius = 10  # amount of pixels that the image has probably (at most) moved by

    ref_im = cropped_edge_array[0]

    move_j_by = [0]
    move_k_by = [0]

    for i in range(1, n):
        brightness_sums = np.zeros((2 * radius, 2 * radius))
        for mj in range(2 * radius):
            for mk in range(2 * radius):
                curr_im = cropped_edge_array[i].copy()
                curr_im = np.roll(curr_im, int(mj - radius), axis=0)
                curr_im = np.roll(curr_im, int(mk - radius), axis=1)
                brightness_sums[mj, mk] = np.sum(np.ravel((curr_im + ref_im) ** 2))  # weigh sum by brightness
        mj, mk = np.unravel_index(brightness_sums.argmax(), brightness_sums.shape)
        if brightness_sums[mj, mk] > brightness_sums[radius, radius]:
            move_j_by.append(mj - radius)
            move_k_by.append(mk - radius)
        elif (np.abs(mj) > 9) and (np.abs(mk) > 9):
            move_j_by.append(99)
            move_k_by.append(99)
        else:
            move_j_by.append(0)
            move_k_by.append(0)

    print 'move_j_by:'
    print move_j_by
    print 'move_k_by:'
    print move_k_by

    cropped_aligned_edges = cropped_edge_array.copy()

    for i in range(n):
        if (move_j_by[i] < 11) and (move_k_by[i] < 11):
            cropped_aligned_edges[i] = np.roll(cropped_aligned_edges[i], int(move_j_by[i]), axis=0)
            cropped_aligned_edges[i] = np.roll(cropped_aligned_edges[i], int(move_k_by[i]), axis=1)

    plt.figure(figsize=(10, 5))
    plt.imshow(np.sum(cropped_aligned_edges, axis=0))
    plt.colorbar()
    # plt.title(folder_name + '\n sum of aligned edges of cropped image')
    plt.show()

    # ## Now move the entire image for all images in the stack

    aligned_image_stack = [image_stack[0]]  # np.zeros(image_stack.shape)
    brightness_list = [np.median(image_stack[0])]  # np.zeros(n)
    new_degrees = [degrees[0]]
    new_background = [background[0]]
    for i in range(1, n):
        if (move_j_by[i] < 11) and (move_k_by[i] < 11):
            curr_im = image_stack[i].copy()
            curr_im = np.roll(curr_im, int(move_j_by[i]), axis=0)
            curr_im = np.roll(curr_im, int(move_k_by[i]), axis=1)

            aligned_image_stack.append(curr_im)  # [i] = curr_im
            brightness_list.append(np.median(curr_im))  # [i] = np.median(curr_im)

            new_degrees.append(degrees[i])
            new_background.append(background[i])

    brightest_index = int(np.argmax(brightness_list))
    shift_angle_by = degrees[brightest_index]
    if n < 20:
        new_degrees = (np.array(new_degrees) - shift_angle_by) % 180.
    else:
        new_degrees = (np.array(new_degrees) - shift_angle_by) % 360.

    # ## Before we save, check that the images are aligned

    aligned_edge_array = np.zeros(np.array(aligned_image_stack).shape)
    for i in range(len(aligned_image_stack)):
        edges = canny(aligned_image_stack[i], sigma=8)  # , sigma=3, low_threshold=9, high_threshold=10
        edges = edges.astype(np.float32)
        aligned_edge_array[i] = edges

    plt.figure(figsize=(18, 9))

    plt.subplot(1, 2, 1)

    plt.imshow(np.sum(edges_stack, axis=0))
    plt.colorbar()
    plt.title('summed edges of unaligned images')

    plt.subplot(1, 2, 2)

    plt.imshow(np.sum(aligned_edge_array, axis=0))
    plt.colorbar()
    plt.title('summed edges of aligned images')

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(np.sum(aligned_edge_array[:, start_j:end_j, start_k:end_k], axis=0))
    plt.colorbar()
    plt.show()

    # # Now we save

    new_degrees = np.roll(new_degrees, -brightest_index, axis=0)
    new_background = np.roll(new_background, -brightest_index, axis=0)
    aligned_image_stack = np.roll(aligned_image_stack, -brightest_index, axis=0)
    # degrees, aligned_image_stack = zip(*sorted(zip(new_degrees, aligned_image_stack)))
    print new_degrees
    radians = np.array(new_degrees) * np.pi / 180.

    np.savez(folder_name + '\\aligned_image_stack', image_stack=aligned_image_stack, degrees=new_degrees,
             radians=radians, background=new_background)
    print 'just saved aligned image stack for '
    print folder_name

    # # Check to see that it worked!

    npzfile = np.load(folder_name + '\\aligned_image_stack.npz')

    aimstack = npzfile['image_stack']

    n, nj, nk = aimstack.shape

    aedstack = np.zeros((n, nj, nk))

    for i in range(n):
        aedstack[i] = canny(aimstack[i], sigma=8)

    plt.figure(figsize=(20, 10))
    plt.imshow(np.sum(aedstack, axis=0))
    plt.colorbar()
    # plt.title(folder_name + '\n sum of aligned edges of saved image stack')
    plt.show()

    print ''


for folder in folder_name_list:
    align_images_in_folder(directory_name + '\\' + folder)
