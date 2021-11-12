from PIL import Image
import glob, os
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import numpy
import read_log_camera as rlc
import os

#### important, make sure YOU dont loose lots of frames

path = r"C:\data\20190716\dropplets\1"


pathsavedaq = path + "\daq.npy"
daq = np.load(pathsavedaq)
pathsavecam = path + "\cam.npy"
cam = np.load(pathsavecam)
pathsave_time_daq = path + "\\timedaq.npy"
timedaq = np.load(pathsave_time_daq)
pathsave_time_cam = path + "\\timecam.npy"
timecam = np.load(pathsave_time_cam)
#pathsavedaq = path + "\daq_treated.npy"
#out = np.load(pathsavedaq)
#pathsavecam = path + "\cam_treated.npy"
#cam = np.load(pathsavecam)
#pathsave_out_final = path + "\\out_final.npy"
#out_final = np.load(pathsave_out_final)



def change_frame (timedaq, daq, timecam) :
    dt = timedaq [1] - timedaq [0]
    T = timecam[1] - timecam[0]
    a = int(T/dt)+1 #to rewrite
    New_timedaq = []
    New_datadaq = []
    for i in range (int(len(timedaq)/a) - 1) :
        New_timedaq.append(timedaq[i*a])
        # median = np.median(daq[i*a : (i+1)*a])
        # if median >= 0.5 :
        #     median = 1
        #     Newdata.append(median)
        New_datadaq.append(daq[i*a])
    return [New_timedaq, New_datadaq]


[New_timedaq, New_datadaq] = change_frame (timedaq,daq, timecam)
#[Newtime_final, Newdata_final] = change_frame (out_final, timedaq, T)



# def coherence_cam_daq (Newdata, Newtime, cam, timecam, T) :
#     Timemeas =  Newtime[-1]
#     m = np.amin([len(Newtime), len(timecam)])
#     a = 0
#     b = 0
#     c = 0
#     for i in range(m):
#         if timecam[i] < Timemeas:
#             if np.abs(timecam[i] - Newtime[i]) < T:
#                 a += Newdata[i]*cam[i]
#                 b += Newdata[i]**2
#                 c += cam[i]**2

#     cohe = a/np.sqrt(b*c)
#     return cohe

# cohe1 = coherence_cam_daq (Newdata, Newtime, cam, timecam, T)
# cohe2 = coherence_cam_daq (Newdata_final, Newtime_final, cam, timecam, T)
# print "coherence", cohe1
# print cohe2


            
    

plt.figure()
plt.plot(timecam, 0.1*(cam-np.mean(cam)), "r.-")
plt.plot(timedaq, 5*(daq-np.mean(daq)), "b.-")
plt.plot(New_timedaq,New_datadaq,"g*-")
plt.show()


# def coherence (timecam, cam, timedaq, out):
    
