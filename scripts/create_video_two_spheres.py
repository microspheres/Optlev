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
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu

from skimage.feature import register_translation

pathlist = [r"C:\data\20200313\two_spheres\15um_german\2\1", ]

see_frame = False
use_threshold = True

# FB on the bottom for march 13
coorL = [33, 25, 105, 95]
coorR = [160, 13, 270, 90]

# # FB on the top
#coorL = [47, 35, 133, 110]
#coorR = [222, 22, 345, 98]

def get_files_path(path):
    file_list = glob.glob(path+"\*.tiff")
    return file_list

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist


def get_sub_image(name, coordinates):
    im = Image.open(name)
    left, top, right, bottom = coordinates
    im = im.crop((left, top, right, bottom))
    return im

def increasing(lenght):
    A = []
    a = 1
    for i in range(lenght):
        A.append(a)
        a = a + 1
    A = np.array(A)*1.
    return A

# this is for plotting the images to check the coordinates crop
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
plt.imshow(imageR)

if see_frame:
    plt.show()
# the crop test ends here


def threshold(imageL, imageR):
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


def positionLR(nameref, name, coorL, coorR):
    imageLref = get_sub_image(nameref, coorL)
    imageRref = get_sub_image(nameref, coorR)
    imageLref = img_as_ubyte(imageLref)
    imageRref = img_as_ubyte(imageRref)

    imageL = get_sub_image(name, coorL)
    imageR = get_sub_image(name, coorR)
    imageL = img_as_ubyte(imageL)
    imageR = img_as_ubyte(imageR)
    
    if use_threshold:
        #imageL, imageR = threshold(imageL, imageR)
        imageLref, imageRref = threshold(imageLref, imageRref)

    shiftL, errorL, diffphaseL = register_translation(imageLref, imageL, upsample_factor = 100)
    shiftR, errorR, diffphaseR = register_translation(imageRref, imageR, upsample_factor = 100)

    sl = shiftL[1]
    sr = shiftR[1]
    slx = shiftL[0]
    srx = shiftR[0]
    
    #print "L", SL
    #print "R", SR

    return [sl, sr, slx, srx]


def positionALL(filelist, coorL, coorR):
    R = []
    L = []
    Rx = []
    Lx = []
    # a = 0
    nameref = filelist[0]
    for i in filelist:
        l, r, lx, rx = positionLR(nameref, i, coorL, coorR)
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

def draw_circle(filename, coorL, coorR, lx, l, rx, r):
    image = Image.open(filename)
    image = img_as_ubyte(image)
    radius = 5 ##15
    a = coorL[0] - l + 35
    b = coorL[1] - lx + 35
    c = coorR[0] - r + 55
    d = coorR[1] - rx + 35
    im = plt.imshow(image)
    ## Hardcoded zoom -- Fernando, fix this!
    #plt.xlim(205, 230)
    #plt.ylim(35, 65)
    ##
    circlel = plt.Circle((a,b), radius, edgecolor = "r", facecolor = "none" )
    circler = plt.Circle((c,d), radius, edgecolor = "r", facecolor = "none" )
    ax = plt.gca()
    c1=ax.add_patch(circler)
    c2=ax.add_patch(circlel)
    return [im, c1, c2]


#plt.figure()
#image = draw_circle(filelist[3], coorL, coorR, 20, 20, 20, 20)
#plt.imshow(image)
#plt.show()

def video_with_circle(filelist, coorL, coorR):

    pathvideo = r"C:\data\20200313\two_spheres\15um_german\video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    video = cv2.VideoWriter(pathvideo, fourcc, 15.0, (332,108), isColor = True)
    for i in filelist:
        
        l, r, lx, rx = positionLR(i, coorL, coorR)
        image = draw_circle(i, coorL, coorR, lx, l, rx, r)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)

    video.release()
    cv2.destroyAllWindows()
    return 


import matplotlib.animation as animation

def video_with_circlePLT(filelist, coorL, coorR):

    pathvideo = r"C:\data\20200313\two_spheres\15um_german\video_long.mp4"

    frames = []
    fig = plt.figure()
    nameref = filelist[0]
    
    for i in filelist:
        
        l, r, lx, rx = positionLR(nameref, i, coorL, coorR)
        image = draw_circle(i, coorL, coorR, lx, l, rx, r)
        frames.append(image)
        
    ani = animation.ArtistAnimation(fig, frames, interval = 50, blit = True, repeat_delay = 1000)
    ani.save(pathvideo)

    plt.show()


video_with_circlePLT(filelist[0:2000], coorL, coorR)
plt.show()

















def positionLR(name, coorL, coorR):
    imageL = get_sub_image(name, coorL)
    imageR = get_sub_image(name, coorR)
    imageL = img_as_ubyte(imageL)
    imageR = img_as_ubyte(imageR)
    
    if use_threshold:
        values = np.linspace(0.05, 1, 20)
        threshL = threshold_otsu(imageL)
        threshR = threshold_otsu(imageR)
        imageLstat = imageL
        imageRstat = imageR
        imageL = np.ones(imageL.shape) # ones instead of zeros otherwise there will be division by zero.
        imageR = np.ones(imageR.shape) 
        for i in values:
            imageLa = imageLstat > i*threshL
            imageRa = imageRstat > i*threshR
            imageL = imageL + imageLa*1.
            imageR = imageR + imageRa*1.
        imageLx = np.rot90(imageL)
        imageRx = np.rot90(imageR)
    else:
        imageLx = np.rot90(imageL)
        imageRx = np.rot90(imageR)

    SL = []
    SLx = []

    for j in imageL:
        L = len(j)
        incL = increasing(L)
        sL = np.sum(j*incL)/np.sum(j)
        SL.append(sL)
    SL = np.sum(SL)

    for j in imageLx:
        Lx = len(j)
        incLx = increasing(Lx)
        sLx = np.sum(j*incLx)/np.sum(j)
        SLx.append(sLx)
    SLx = np.sum(SLx)

    SR = []
    SRx = []

    for j in imageR:
        R = len(j)
        incR = increasing(R)
        sR = np.sum(j*incR)/np.sum(j)
        SR.append(sR)
    SR = np.sum(SR)

    for j in imageRx:
        Rx = len(j)
        incRx = increasing(Rx)
        sRx = np.sum(j*incRx)/np.sum(j)
        SRx.append(sRx)
    SRx = np.sum(SRx)
    
    #print "L", SL
    #print "R", SR

    return [SL -2550, SR-4500, SLx-2700, SRx-4100]
    return [SL -2584, SR-4225, SLx-2477, SRx-4340]
