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

pathlist = [r"C:\data\20200313\two_spheres\15um_german\1\1", r"C:\data\20200313\two_spheres\15um_german\1\2", r"C:\data\20200313\two_spheres\15um_german\1\3", r"C:\data\20200313\two_spheres\15um_german\1\4", ]

pathlist = glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\2", "*", ""))
pathlist = pathlist + glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\3", "*", ""))
pathlist = pathlist + glob.glob(os.path.join(r"C:\data\20200311\two_spheres\15um_german\4", "*", ""))

see_frame = False
use_threshold = False

# FB on the bottom for march 13
#coorL = [33, 25, 105, 95]
#coorR = [160, 13, 270, 90]

# # FB on the top
coorL = [47, 35, 133, 110]
coorR = [222, 22, 345, 98]

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

path = pathlist[0]
filelist = get_files_path(path)
filelist = list_file_time_order(filelist)
imageL = get_sub_image(filelist[400], coorL)
imageR = get_sub_image(filelist[400], coorR)
imageL = img_as_ubyte(imageL)
imageR = img_as_ubyte(imageR)
if use_threshold:
    values = np.linspace(0.05, 1, 20)
    threshL = threshold_otsu(imageL)
    threshR = threshold_otsu(imageR)
    imageLstat = imageL
    imageRstat = imageR
    imageL = np.ones(imageL.shape)
    imageR = np.ones(imageR.shape)
    for i in values:
        imageLa = imageLstat < i*threshL
        imageRa = imageRstat < i*threshR
        imageL = imageL + imageLa*1.
        imageR = imageR + imageRa*1.

plt.figure()
plt.imshow(imageL)

plt.figure()
plt.imshow(imageR)

if see_frame:
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
            imageLa = imageLstat < i*threshL
            imageRa = imageRstat < i*threshR
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
    SL = sum(SL)

    for j in imageLx:
        Lx = len(j)
        incLx = increasing(Lx)
        sLx = np.sum(j*incLx)/np.sum(j)
        SLx.append(sLx)
    SLx = sum(SLx)

    SR = []
    SRx = []

    for j in imageR:
        R = len(j)
        incR = increasing(R)
        sR = np.sum(j*incR)/np.sum(j)
        SR.append(sR)
    SR = sum(SR)

    for j in imageRx:
        Rx = len(j)
        incRx = increasing(Rx)
        sRx = np.sum(j*incRx)/np.sum(j)
        SRx.append(sRx)
    SRx = sum(SRx)
    
    # print "L", SL
    # print "R", SR

    return [SL, SR, SLx, SRx]

def positionALL(filelist, coorL, coorR):
    R = []
    L = []
    Rx = []
    Lx = []
    # a = 0
    for i in filelist:
        l, r, lx, rx = positionLR(i, coorL, coorR)
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


for j in pathlist:
    path = j
    filelist = get_files_path(path)
    filelist = list_file_time_order(filelist)

    L, R, Lx, Rx = positionALL(filelist, coorL, coorR)
    L = L - np.mean(L)
    R = R - np.mean(R)
    Lx = Lx - np.mean(Lx)
    Rx = Rx - np.mean(Rx)

    save = [L, R, Lx, Rx]
    if use_threshold:
        pathsave = path + "\\timestream_with_threshold.npy"
    else:
        pathsave = path + "\\timestream.npy"
    np.save(pathsave, save)
    print j

plt.figure()
plt.plot(L)
plt.plot(R)


Lpsd, freqs = matplotlib.mlab.psd(L, Fs = 450, NFFT = 2**11)
Rpsd, freqs = matplotlib.mlab.psd(R, Fs = 450, NFFT = 2**11)
Lpsdx, freqs = matplotlib.mlab.psd(Lx, Fs = 450, NFFT = 2**11)
Rpsdx, freqs = matplotlib.mlab.psd(Rx, Fs = 450, NFFT = 2**11) 
    
plt.figure()
plt.loglog(freqs, Lpsd)
plt.loglog(freqs, Rpsd)

plt.figure()
plt.loglog(freqs, Lpsdx, label = "leftx")
plt.loglog(freqs, Rpsdx)
plt.legend()


plt.show()













    

# from skimage import data, color, draw
# from skimage.transform import hough_circle, hough_circle_peaks
# from skimage.feature import canny
# from skimage.draw import circle_perimeter
# from skimage.util import img_as_ubyte

# from skimage.feature import register_translation
# from skimage.feature.register_translation import _upsampled_dft
# from scipy.ndimage import fourier_shift
# from skimage.filters import threshold_otsu

# from scipy.optimize import curve_fit

# image = get_image_file(filelist, 100)

# image = img_as_ubyte(image)

# # thresh = threshold_otsu(image)
# # image = image < .3*thresh
# # image = image*1.
# # plt.figure()
# # plt.imshow(image)
# # plt.show()


# edges = canny(image, sigma=3, low_threshold=40, high_threshold=45)

# hough_radii = np.arange(18, 22, 2)
# hough_res = hough_circle(edges, hough_radii)

# accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=2)

# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
# image = color.gray2rgb(image)
# for center_y, center_x, radius in zip(cy, cx, radii):
#     circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
#     image[circy, circx] = (220, 20, 20)

# ax.imshow(image, cmap=plt.cm.gray)
# plt.show()










# def func((x, y),x0, y0, x1, y1):

#     I0 = 100.*np.exp(-0.0005*(x-x0)**2. - 0.0005*(y-y0)**2.)

#     I1 = 100.*np.exp(-0.0005*(x-x1)**2. - 0.0005*(y-y1)**2.)

#     g = 100 - I0 - I1
    
#     return g


# X = np.linspace(0, image.shape[1], image.shape[1])
# Y = np.linspace(0, image.shape[0], image.shape[0])
# X, Y = np.meshgrid(X, Y)



# # data = func((X, Y), 88, 250., 162., 258.)
# # #data = func2((X, Y), 88, 250 , 162, 258, 20, 20, 10, 3)

# # plt.figure()
# # plt.imshow(data.reshape(492, 492))
# # plt.colorbar()
# # plt.show()


# xdata = np.vstack((X.ravel(), Y.ravel()))
# p0 = [88, 250., 162., 258.]

# popt, pcov = curve_fit(func, xdata, image.ravel(), p0 = p0, maxfev = 1000)

# data = func((X, Y), *popt)

# print popt

# plt.figure()
# plt.imshow(data.reshape(492, 492))
# # plt.imshow(image)
# plt.colorbar()
# plt.show()



























#p0 = [88, 250 , 162, 258, 15, 15]
# def func2((x, y), x0, y0, x1, y1, r0, r1):

#     A0i = (x - x0)**2 + (y - y0)**2 - r0**2
#     N0i = np.zeros(A0i.shape)
#     for i in range(len(x)):
#         for j in range(len(y)):
#             if A0i[i,j] > 0:
#                 N0i[i,j] = 0
#             else:
#                 N0i[i,j] = 1
                
#     A0o = (x - x0)**2 + (y - y0)**2 - (r0-4)**2
#     N0o = np.zeros(A0o.shape)
#     for i in range(len(x)):
#         for j in range(len(y)):
#             if A0o[i,j] > 0:
#                 N0o[i,j] = 1
#             else:
#                 N0o[i,j] = 0

#     A1i = (x - x1)**2 + (y - y1)**2 - r1**2
#     N1i = np.zeros(A1i.shape)
#     for i in range(len(x)):
#         for j in range(len(y)):
#             if A1i[i,j] > 0:
#                 N1i[i,j] = 0
#             else:
#                 N1i[i,j] = 1
                
#     A1o = (x - x1)**2 + (y - y1)**2 - (r1-4)**2
#     N1o = np.zeros(A0o.shape)
#     for i in range(len(x)):
#         for j in range(len(y)):
#             if A1o[i,j] > 0:
#                 N1o[i,j] = 1
#             else:
#                 N1o[i,j] = 0
    
#     return ((N0i+N0o) + N1i + N1o)-2

# def func3((x, y), x0, y0, x1, y1, r0, r1):

#     A0i = (x - x0)**2 + (y - y0)**2 - r0**2
#     N0i = np.zeros(A0i.shape)
#     for i in range(len(x*y)):
#         if A0i[i] > 0:
#             N0i[i] = 0
#         else:
#             N0i[i] = 1
            
#     A0o = (x - x0)**2 + (y - y0)**2 - (r0-4)**2
#     N0o = np.zeros(A0o.shape)
#     for i in range(len(x*y)):
#         if A0o[i] > 0:
#             N0o[i] = 1
#         else:
#             N0o[i] = 0

#     A1i = (x - x1)**2 + (y - y1)**2 - r1**2
#     N1i = np.zeros(A1i.shape)
#     for i in range(len(x*y)):
#         if A1i[i] > 0:
#             N1i[i] = 0
#         else:
#             N1i[i] = 1
            
#     A1o = (x - x1)**2 + (y - y1)**2 - (r1-4)**2
#     N1o = np.zeros(A1o.shape)
#     for i in range(len(x*y)):
#         if A1o[i] > 0:
#             N1o[i] = 1
#         else:
#             N1o[i] = 0
    
#     return (N0i+N0o + N1i+N1o)-2
