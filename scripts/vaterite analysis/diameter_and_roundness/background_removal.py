import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import join
from scipy.misc import imread
# from scipy.ndimage import gaussian_filter
from skimage.io import imread  # , imsave

# from tifffile import imwrite

path_name = r'C:\Users\Sumita\Documents\Research\Microspheres\vaterite spheres\2019-05-09'

folder_name_list = ['alumina',
                    'corpuscular_15um_silica',
                    'gadi_vaterite',
                    'german_8um_vaterite',
                    'german_11um_vaterite',
                    'german_22um_silica_10X',
                    'german_22um_silica_40X'
]

blank = np.load('blank.npy')
blank_bg = float(np.median(blank))


def remove_background(folder_name):
    imagepath = path_name + '\\' + folder_name
    imagelist = glob(join(imagepath, '*.png'))
    imagelist.extend(glob(join(imagepath, '*.jpg')))

    for imname in imagelist:
        print imname

        im = imread(imname, flatten=True)

        im_bg = float(np.median(im))
        im = im * blank_bg / im_bg

        new_im = blank - im
        new_im = new_im - np.median(new_im)
        new_im[new_im < 0] = 0
        new_im = new_im - np.median(new_im)
        new_im[new_im < 0] = 0
        new_im = new_im * 400. / np.max(new_im)
        new_im[new_im > 255] = 255

        # new_im2 = 255 - new_im.copy()
        # new_im = gaussian_filter(new_im2, sigma=5)

        plt.imsave(imname[:imname.rfind('.')] + '_cleaned.png', 255 - new_im, cmap='gray')


for folder in folder_name_list:
    remove_background(folder)
