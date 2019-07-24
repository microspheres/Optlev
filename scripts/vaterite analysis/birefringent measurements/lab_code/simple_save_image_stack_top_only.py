import numpy as np
from glob import glob
from os.path import join
from scipy.misc import imread

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
default_path = default_path + '2019-05-09\\'
extension = ".png"
folder_names = [#'alumina_birefringence',
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


def save_image_stack(folder_name):
    path_name = default_path + folder_name
    file_list = glob(join(path_name, '*.png'))
    file_list.extend(glob(join(path_name, '*.jpg')))

    image_stack = []  # original image
    degrees = []  # in degrees
    radians = []
    background = []
    for file_name in file_list:
        im = imread(file_name, flatten=True)
        image_stack.append(im)
        background.append(np.median(im))
        d = int(file_name[file_name.rfind('\\') + 1:file_name.rfind('.')])
        degrees.append(d)
        radians.append(d * np.pi / 180.)

    degrees, radians, image_stack = zip(*sorted(zip(degrees, radians, image_stack)))

    np.savez(path_name + '\\image_stack.npz', image_stack=image_stack, degrees=degrees, radians=radians, background=background)


for f in folder_names:
    print 'saving image stack in ' + f
    save_image_stack(f)
