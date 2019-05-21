from PIL import Image
import glob, os
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import numpy


path = r"C:\data\20190517\dropplets\11"

pathframe = r"C:\data\20190517\dropplets\11\frames_120Hz"

fname = "data.avi"



def from_video_to_image(name): # fix the path in which is saved, name is the path+file name
    cap = cv2.VideoCapture(name)

    success,image = cap.read()
    count = 0
    success = True
    while success:

        cv2.imwrite(r'C:\data\20190517\dropplets\11\frames_120Hz\\' + "frame%d.jpg" % count, image)   # save frame as JPEG file
        success,image = cap.read()
        print 'Read a new frame: ', success
        count += 1

# from_video_to_image(r"C:\data\20190517\dropplets\11\flow_120Hz_(300_-900)mbar.avi")

def get_image_file(name):
    im = Image.open(name)
    plt.imshow(im)
    return im

# get_image_file(r"C:\data\20190517\dropplets\11\frames_120Hz\frame927.jpg")

def get_pixel_file(name):
    image = get_image_file(name)
    px = image.load()

    width, height = image.size
    pixel = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            pixel[i,j] = px[i,j][0]

    pixel_signal = np.sum(pixel[384,14:24])
    pixel_noise = np.sum(pixel[384,0:10])
    pixel = pixel_signal - pixel_noise
    return pixel

def get_files_path(path):
        file_list = glob.glob(path+"\*.jpg")
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

# name = os.path.join(path, r"data1.h5")

# daq = getdata(name)

# x = np.array(range(len(daq)))*0.001

file_list = get_files_path(pathframe)

a = get_all(file_list)

xframe = np.array(range(len(a)))*(1/120.)

plt.figure()
plt.plot(xframe,a-np.mean(a))
# plt.plot(x,500*daq-np.mean(500*daq))


plt.show()
