from PIL import Image
import glob, os
import cv2
import matplotlib.pyplot as plt
import h5py
import numpy as np
import numpy


# path = r"C:\data\20190524\droplets\test_log"

def get_files_path(path):
        file_list = glob.glob(path+"\*txt")
        return file_list

def get_lines(path):
    file_list = get_files_path(path)
    
    f = open(file_list[0], "r")

    Lines = []
    
    for i in f:
        Lines.append(i)

    return Lines


# a = get_lines(path)[0].split("-")[1]
# b = a.split(":")
# print 3600.0*np.float(b[0]) + 60.0*np.float(b[1]) + np.float(b[2])
# print len(get_lines(path)[0].split("-")[2].split("received"))

def get_times(path):
    lines = get_lines(path)

    aux1 = lines[0].split("-")[1]
    aux2 = aux1.split(":")
    
    t0 = 3600.0*np.float(aux2[0]) + 60.0*np.float(aux2[1]) + np.float(aux2[2])

    t = []
    
    for i in lines:
        if len(i.split("-")[2].split("received")) == 2:
            #print i.split("-")[2]
            auxf = i.split("-")[1]
            auxf = auxf.split(":")
            tauxf = 3600.0*np.float(auxf[0]) + 60.0*np.float(auxf[1]) + np.float(auxf[2])
            t.append(tauxf)

    t = np.array(t)
    t = t - t0

    return t

# b = get_times(path)


# a = np.arange(0,len(b)*0.02,0.02)
# print a

# plt.figure()
# plt.plot(a, b)
# plt.xlabel("True clock")
# plt.ylabel("Frame aquisition time")
# plt.grid()
# plt.show()
