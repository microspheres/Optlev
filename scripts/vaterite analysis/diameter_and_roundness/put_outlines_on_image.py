import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import join
from scipy.misc import imread
from skimage.color import gray2rgb
from skimage.io import imread

path_name = r'C:\Users\Sumita\Documents\Research\Microspheres\vaterite spheres\2019-05-09'

folder_name_list = ['alumina',
                    'corpuscular_15um_silica',
                    'gadi_vaterite',
                    'german_8um_vaterite',
                    'german_11um_vaterite',
                    'german_22um_silica_10X',
                    'german_22um_silica_40X'
                    ]


def add_outlines(folder_name):
    imagepath = path_name + '\\' + folder_name
    print imagepath
    imagelist = glob(join(imagepath, 'outline*'))
    print len(imagelist)

    for imname in imagelist:
        print imname
        outim = imread(imname, flatten=True)

        try:
            imnum = int(imname[imname.rfind('outline') + 7:imname.rfind('.')])
        except:
            imnum = int(imname[imname.rfind('outline') + 8:imname.rfind('.')])

        try:
            og_im_name = imagepath + '\\' + str(imnum) + '.png'
            og_im = imread(og_im_name, flatten=True)
        except:
            og_im_name = imagepath + '\\' + str(imnum) + '.jpg'
            og_im = imread(og_im_name, flatten=True)

        toadd = np.array(outim, dtype=bool)
        toadd = np.bitwise_xor(np.ones(toadd.shape, dtype=bool), toadd)

        # Draw the most prominent 5 circles
        image2 = gray2rgb(og_im)
        image2[toadd] = (5, 250, 250)

        plt.imsave(og_im_name[:og_im_name.rfind('.')] + '_outlined.png', image2)


for folder in folder_name_list:
    add_outlines(folder)
