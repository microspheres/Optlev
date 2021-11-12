from PIL import Image
import glob, os
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import numpy
import read_log_camera as rlc
import os
from scipy.signal import find_peaks


path = r"C:\data\20190716\dropplets\8"

pathframe = r"C:\data\20190716\dropplets\8\frames_300Hz"

fname = "video.avi"

log_camera = True

make_frame = False

see_frame = False




def from_video_to_image(name): # fix the path in which is saved, name is the path+file name
    cap = cv2.VideoCapture(name)

    success,image = cap.read()
    count = 0
    success = True
    while success:

        cv2.imwrite(pathframe+'\\' + "frame%d.jpg" % count, image)   # save frame as JPEG file
        success,image = cap.read()
        print 'Read a new frame: ', success
        count += 1
        
if make_frame:
    from_video_to_image(path+'\\'+"video.avi")


    

def get_image_file(name):
    im = Image.open(name)
    if see_frame:
        plt.imshow(im)
    return im

if see_frame:
    get_image_file(pathframe+'\\'+"frame1271.jpg")




def get_pixel_file(name):
    image = get_image_file(name)
    px = image.load()

    width, height = image.size
    pixel = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            pixel[i,j] = px[i,j][0]

    #pixel_signal = np.sum(pixel[170:190,16:23])
    #ref_pixel = pixel[346,37]
    pixel_signal = np.sum(pixel[195:196,36:37])
    #pixel_noise = np.sum(pixel[300:301,0:2])

    pixel = pixel_signal 
    return pixel_signal


def list_file_time_order(filelist):
    filelist.sort(key = os.path.getmtime)
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
	return x, 1./Fs




    
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


def butter_highpass(highcut, fs, order):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='high')
    return b, a

def butter_highpass_filter(data, highcut, fs, order):
    b, a = butter_highpass(highcut,  fs, order=order)
    y = filtfilt(b, a, data)
    return y






def treat_daq_signal(daq, Filter): # corrects the filtering of the lockinamp
    out = []
    mean = np.mean(daq)
    ss = np.std(daq)
    threshold = mean + ss
    for i in daq:
        if i > threshold :
            aux = 1
            out.append(aux)
        else:
            aux = 0
            out.append(aux)

    out = np.array(out)

    if Filter:
        out = butter_bandpass_filter(out, 0.0001, 60, 1000, 2)
    return out


def treat_image_signal(data):
    ref_pixel = np.mean(data)
    ss = np.std(data)
    aux1 = np.zeros(len(data))
    aux2 = np.zeros(len(data))
    c1 = []
    c2 = []
    for i in range(len(data)) :
        if data[i] < ref_pixel - ss:
            aux1[i] = -1
            c1.append(i)
        if data[i] > ref_pixel + ss:
            aux2[i] = 1
            c2.append(i)
    aux = aux1+aux2

    # plt.figure()
    # plt.plot([i for i in range(len(data))],aux)
    # plt.show()
    
    kbis = 0
    c = 0 #counts the number of dropplets
    
    for i in c2 :
        if i >= kbis :
            
            j = i
            while j>=0 and aux[j]>=0 :
                j = j-1
            jbis = j
            while jbis>=0 and aux[jbis]<0 :
                jbis = jbis-1
                
            k = i
            while k<len(data) and aux[k]>=0 :
                k = k+1
            kbis = k
            while kbis<len(data) and aux[kbis]<0 :
                kbis = kbis + 1
                
            low_ind = j - int((j-jbis)/2)
            high_ind = k + int((kbis-k)/2) 

            for l in range(low_ind,high_ind) :
                aux[l] = 1
                
            c = c+1

    aux_final = aux        
    for i in range (len(data)):
        if aux [i]<0 :
            aux_final[i] = 0      
    print 'Number of dropplets=', c
    
    return aux_final

# def treat_out (out) :
#     out_final = out
#     Iout = 1.*(np.sum(out[i] for i in range (len(out))))/len(out)
#     Icam = 1.*(np.sum(cam[i] for i in range (len(cam))))/len(cam)
#     print Iout,len(out), Icam, len (cam)
#     r = len(out)/len(cam)
#     p = Icam/Iout
#     q = int(p/2)+1
#     print p
#     print int(p/2)+1
#     print r
#     print r*q
    

#     k = 0
#     for i in range (len(out)) :
#         if i>=k :
#             if out[i] == 1 :
#                 if i>r*q and i<len(out)-r*q:
#                     if out[i-1] == 0 :
#                         print "m=",i
#                         for j in range (i-r*q, i) :
#                             print "mm=",j
#                             out_final[j] = 1
#                             print out_final[j]
#                     if out[i+1] == 0 :
#                         print "p=", i
#                         for j in range(i, i+r*q) :
#                             print "pp=",j
#                             out_final[j] = 1
#                             print out_final[j]
#                         k = i+r*q
#     return out_final


def treat_out2(data):
    ze = np.where(data == 0)[0]
    one = np.where(data == 1)[0]
    # amazing, aux = data creates massive changes in the data that is input to this function
    aux = []
    for i in data:
        aux.append(i)
    aux = np.array(aux)
    a = int(45)
    for i in one:
        if i > a:
            if i - 1 in ze:
                for j in range (i-a, i) :
                    aux[j] = 1
        if i < len(aux) - a:
            if i + 1 in ze:
                for j in range(i, i + a):
                    aux[j] = 1
    return aux
                





if not make_frame and not see_frame :
    name = os.path.join(path, r"data1.h5")

    daq,Ts = getdata(name)
    x = np.array(range(len(daq)))*Ts

    file_list = get_files_path(pathframe)
    cam = get_all(file_list)
    
    if log_camera:
        xframe = rlc.get_times(path)
    else:
        frate = 50.
        xframe = np.array(range(len(a)))/frate


    out = treat_daq_signal(daq, True)
    outbis = treat_daq_signal(out, False)
    cam_treated = treat_image_signal(cam)
    out_final = treat_out2(outbis)


    pathsavedaq = path + "\daq"
    np.save(pathsavedaq, daq)
    pathsavecam = path + "\cam"
    np.save(pathsavecam, cam)
    pathsave_time_daq = path + "\\timedaq"
    np.save(pathsave_time_daq, x)
    pathsave_time_cam = path + "\\timecam"
    np.save(pathsave_time_cam, xframe)
    #pathsavedaq = path + "\daq_treated"
    #np.save(pathsavedaq, out)
    #pathsavecam = path + "\cam_treated"
    #np.save(pathsavecam, cam)
    #pathsave_out_final = path + "\\out_final"
    #np.save(pathsave_out_final, out_final)


    
    plt.figure()
    plt.plot(xframe,(cam-np.mean(cam)), ".")
    plt.plot(x,800.*(daq-np.mean(daq)))
    #plt.plot(xframe, 100*cam-50)
    #plt.plot(x, 100*out,"k-")
    #plt.plot(x, 200*out_final,"g-")

    
    plt.xlabel("Time [s]")
    plt.ylabel("Arb Units")
    plt.grid()



    # plt.figure()
    # plt.plot(xframe,a, ".")
    # plt.plot(x, 100*out,"k-")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Arb Units")
    #plt.grid()

    # b = rlc.get_times(path)
    # a = np.arange(0,len(b)*0.02,0.02)
    # plt.figure()
    # plt.plot(a, b)
    # plt.xlabel("True clock")
    # plt.ylabel("Frame aquisition time")
    # plt.grid()

    
        
plt.show()
