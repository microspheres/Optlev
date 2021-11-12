from PIL import Image
import glob, os
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import numpy, matplotlib
import read_log_camera as rlc
import os
from scipy.signal import find_peaks
from skimage.util import img_as_ubyte, crop
from skimage.filters import threshold_otsu
from skimage.feature import register_translation
from skimage.transform import rotate
import multiprocessing as mp
import time

pathlist = [r"C:\data\20200313\two_spheres\15um_german\2\only_1_sphere",  ]

#pathlist = glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\3", "*", ""))
#pathlist = pathlist + glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\3", "*", ""))
#pathlist = pathlist + glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\4", "*", ""))

#pathlist = [r"C:\data\20200313\two_spheres\15um_german\2\1"]
#print pathlist
see_frame = False
use_threshold = True # always set True

use_rotation = True

# FB on the bottom for march 13
coorL = [33-12, 25-12, 105+12, 95+12]
coorR = [160-12, 13-12, 270+12, 90+12]

# # march 11
#coorL = [47, 35, 133, 110]
#coorR = [222, 22, 345, 98]

upsample = 100 # the result looks the same with upsample = 1000

def get_files_path(path):
    file_list = glob.glob(path+"\*.tiff")
    return file_list

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


def get_sub_image(name, coordinates): # this returnd the cropped images
    im = Image.open(name)
    left, top, right, bottom = coordinates
    im = im.crop((left, top, right, bottom))
    return im

def rot_special(image, angle, N_crop):
    image = rotate(image*1.0, angle*1.0)
    image = image[N_crop:len(image) - N_crop]
    image = np.rot90(image)
    image = image[N_crop:len(image) - N_crop]
    image = np.rot90(image)
    image = np.rot90(image)
    image = np.rot90(image)
    return image


# this is for displaying the image cropped
path = pathlist[0]
filelist = get_files_path(path)
filelist = list_file_time_order(filelist)
imageL = get_sub_image(filelist[400], coorL)
imageR = get_sub_image(filelist[400], coorR)
imageL = img_as_ubyte(imageL)
imageR = img_as_ubyte(imageR)
if use_threshold:
    values = np.linspace(0.05, 0.5, 15)
    threshL = threshold_otsu(imageL)
    threshR = threshold_otsu(imageR)
    imageLstat = imageL
    imageRstat = imageR
    imageL = np.ones(imageL.shape)
    imageR = np.ones(imageR.shape)
    for i in values:
        imageLa = imageLstat > i*threshL
        imageRa = imageRstat > i*threshR
        imageL = imageL + imageLa*1.
        imageR = imageR + imageRa*1.

plt.figure()
plt.imshow(imageL)

plt.figure()
if use_rotation:
    imageR = rot_special(imageL, 0, 16)
plt.imshow(imageR)

if see_frame:
    plt.show()
# this ends the crop view


def threshold(imageL, imageR): # returns 2 iamages with threshold
    values = np.linspace(0.05, 0.5, 15)
    threshL = threshold_otsu(imageL)
    threshR = threshold_otsu(imageR)
    imageLstat = imageL
    imageRstat = imageR
    imageL = np.zeros(imageL.shape)
    imageR = np.zeros(imageR.shape) 
    for i in values:
        imageLa = imageLstat > i*threshL
        imageRa = imageRstat > i*threshR
        imageL = imageL + imageLa*1.
        imageR = imageR + imageRa*1.
    return [imageL, imageR]


def positionLR(nameref, name, coorL, coorR, angle): # outputs the sphere position for each frame name
    imageLref = get_sub_image(nameref, coorL)
    imageRref = get_sub_image(nameref, coorR)
    imageLref = img_as_ubyte(imageLref)
    imageRref = img_as_ubyte(imageRref)

    imageL = get_sub_image(name, coorL)
    imageR = get_sub_image(name, coorR)
    imageL = img_as_ubyte(imageL)
    imageR = img_as_ubyte(imageR)

    if use_rotation:
        imageLref = rot_special(imageLref, angle, 16)
        imageRref = rot_special(imageRref, angle, 16)
        imageL = rot_special(imageL, angle, 16)
        imageR = rot_special(imageR, angle, 16)
    
    if use_threshold: # i think it is alwasys good that the ref image is thresholded, that is why the comment in the line below
        #imageL, imageR = threshold(imageL, imageR)
        imageLref, imageRref = threshold(imageLref, imageRref)

    shiftL, errorL, diffphaseL = register_translation(imageLref, imageL, upsample_factor = upsample)
    shiftR, errorR, diffphaseR = register_translation(imageRref, imageR, upsample_factor = upsample)

    sl = shiftL[1]
    sr = shiftR[1]
    slx = shiftL[0]
    srx = shiftR[0]
    
    #print "L", SL
    #print "R", SR

    return [sl, sr, slx, srx]


def positionALL(filelist, coorL, coorR, angle): # uses positionLR to generate a timestream for a filelits of images
    R = []
    L = []
    Rx = []
    Lx = []
    # a = 0
    nameref = filelist[0]
    for i in filelist:
        l, r, lx, rx = positionLR(nameref, i, coorL, coorR, angle)
        R.append(r)
        L.append(l)
        Rx.append(rx)
        Lx.append(lx)
        # print a
        # a = a + 1
    R = np.array(R)
    L = np.array(L)
    Rx = np.array(Rx)
    Lx = np.array(Lx)
    return [L, R, Lx, Rx]

def calculate_path(path, angle): # gets a folder, generate a filelist and saves the timestream
    filelist = get_files_path(path)
    filelist = list_file_time_order(filelist)
        
    L, R, Lx, Rx = positionALL(filelist, coorL, coorR, angle)
    L = L - np.mean(L)
    R = R - np.mean(R)
    Lx = Lx - np.mean(Lx)
    Rx = Rx - np.mean(Rx)

    save = [L, R, Lx, Rx]
    if use_threshold:
        if use_rotation:
            pathsave = path + "\\timestream_with_threshold_" + str(angle) + ".npy"
        else:
            pathsave = path + "\\timestream_with_threshold.npy"
    else:
        pathsave = path + "\\timestream.npy"
        
    np.save(pathsave, save)
        
    print path
    seconds = time.time()
    local_time = time.ctime(seconds)
    print "time saved: ", local_time

    return []


angles = [-16, -12, -8, -4, 0, 4, 8, 12, 16]

for i in angles:
    calculate_path(pathlist[0], i)

#for i in pathlist:
#    calculate_path(i)

#if __name__ == '__main__':
    
#    mp.freeze_support()
#    pool = mp.Pool(processes = 1)
#    pool.map(calculate_path, pathlist)
