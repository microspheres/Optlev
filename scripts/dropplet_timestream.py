from PIL import Image
import glob, os
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import numpy
import read_log_camera as rlc
import os


path = r"C:\data\20190605\droplets_salt\1"

pathframe = r"C:\data\20190605\droplets_salt\1\frames_100Hz"

fname = "video.avi"

log_camera = True

def from_video_to_image(name): # fix the path in which is saved, name is the path+file name
    cap = cv2.VideoCapture(name)

    success,image = cap.read()
    count = 0
    success = True
    while success:

        cv2.imwrite(r'C:\data\20190605\droplets_salt\1\frames_100Hz\\' + "frame%d.jpg" % count, image)   # save frame as JPEG file
        success,image = cap.read()
        print 'Read a new frame: ', success
        count += 1

#from_video_to_image(r"C:\data\20190605\droplets_salt\1\video.avi")

def get_image_file(name):
    im = Image.open(name)
    #plt.imshow(im)
    return im

#get_image_file(r"C:\data\20190605\droplets_salt\1\frames_100Hz\frame39.jpg")

def get_pixel_file(name):
    image = get_image_file(name)
    px = image.load()

    width, height = image.size
    pixel = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            pixel[i,j] = px[i,j][0]

    # pixel_signal = np.sum(pixel[290,15:20])
    # pixel_noise = np.sum(pixel[290,0:5])

    pixel_signal = np.sum(pixel[300-1:300+1,35:40])
    pixel_noise = np.sum(pixel[300-1:300+1,0:5])

    pixel = pixel_signal - pixel_noise
    return pixel

def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

def get_files_path(path):
        file_list = glob.glob(path+"\*.jpg")
        file_list = list_file_time_order(file_list)
        return file_list
    
def get_all(file_list):
    PX = []

    for i in file_list:
        px = get_pixel_file(i)
        PX.append(px)
    return PX


def getdata(fname):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
                pid = dset.attrs['PID']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

	x = dat[:, 0]
	return x


    
from scipy.signal import butter, lfilter, filtfilt
from scipy import signal
import scipy.signal as sp

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def treat_daq_signal(daq, threshould, Filter): # corrects the filtering of the lockinamp
    out = []
    for i in daq:
        if i > threshould:
            aux = 1
            out.append(aux)
        else:
            aux = 0
            out.append(aux)


    out = np.array(out)
    if Filter:
        out = butter_bandpass_filter(out, 0.0001, 10, 1000, 2)
    return out
    

    

name = os.path.join(path, r"data1.h5")

daq = getdata(name)

x = np.array(range(len(daq)))*0.001

file_list = get_files_path(pathframe)

a = get_all(file_list)

if log_camera:

    xframe = rlc.get_times(path)
    
else:

    frate = 50.
    
    xframe = np.array(range(len(a)))/frate


out = treat_daq_signal(daq, 0.5, True)
out = treat_daq_signal(out, 0.1, False)
    
plt.figure()
plt.plot(xframe,a-np.mean(a), ".")
plt.plot(x,200.*(daq-np.mean(daq)))
plt.plot(x, 100*out - 100*np.mean(out),"k-")
plt.xlabel("Time [s]")
plt.ylabel("Arb Units")
plt.grid()

plt.figure()
plt.plot(xframe,a, ".")
plt.plot(x, 100*out,"k-")
plt.xlabel("Time [s]")
plt.ylabel("Arb Units")
plt.grid()



b = rlc.get_times(path)


a = np.arange(0,len(b)*0.02,0.02)

plt.figure()
plt.plot(a, b)
plt.xlabel("True clock")
plt.ylabel("Frame aquisition time")
plt.grid()
plt.show()
