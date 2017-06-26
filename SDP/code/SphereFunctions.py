
# coding: utf-8

# In[ ]:

import skimage
import scipy
import fnmatch
import os
from scipy import ndimage
from skimage import io, segmentation, morphology, measure
from skimage import color, data, img_as_float
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
from EquilibriumValuesFromSphereDiameter import getPowerFromSphereSize as gp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

fors = "microscope"
scale = 4.3*1.15
convfac = 6.2/scale #remember to update when switching calibrations
baseimg = "/home/adam/Optlev/SDP/images/" + fors + "/"
basedata = "/home/adam/Optlev/SDP/data/" + fors + "/"
baseresult = "/home/adam/Optlev/SDP/results/" + fors + "/"

def getPaths(num, beta):
    pngpath = baseimg + str(num) + ".png"
    npypath = basedata + str(num) + ".npy"
    txtpath = baseresult + str(num) + " " + str(beta) + ".txt"
    respath = baseresult + str(num) + " " + str(beta) + ".png"
    ovpath = baseresult + str(num) + " " + str(beta) + "ov.png"
    diampath = baseresult + str(num) + "diams.txt"
    return pngpath, npypath, txtpath, respath, ovpath, diampath

def getFiles(num, beta):
    pngpath, npypath, txtpath, respath, ovpath, diampath = getPaths(num, beta)
    image = io.imread(pngpath)
    y, x = np.load(npypath).T
    markers = np.zeros(image.shape, dtype=np.int)
    markers[x.astype(np.int), y.astype(np.int)] = np.arange(len(x))+1
    #markers = morphology.dilation(markers, morphology.disk(15))
    rw = segmentation.random_walker(image, markers, beta=beta)
    properties = measure.regionprops(rw)
    areas = np.zeros(len(x))
    for i in range(0,len(x)):
        areas[i] = properties[i].area
    diameters = 2*np.sqrt(areas/np.pi)/scale
    #rwSphere = getSphere(rw)
    #ov = overlaySeg(image, rwSphere)
    np.savetxt(txtpath, diameters, fmt="%1.8f")
    scipy.misc.imsave(respath, rw)
    #scipy.misc.imsave(ovpath, ov)
    #return rw, image
    

def overlaySeg(image, rw):
    rows, cols = image.shape
    rw_blank = np.zeros(rw.shape)
    # Construct RGB version of grey-level image
    image_color = np.dstack((image, image, image))
    rw_color = np.dstack((rw*255, rw_blank, rw_blank))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(image_color)
    rw_hsv = color.rgb2hsv(rw_color)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = rw_hsv[..., 0]
    img_hsv[..., 1] = rw_hsv[..., 1]

    ov = color.hsv2rgb(img_hsv)
    return ov

#f, axarr = plt.subplots(2, 2)
#axarr[0, 0].imshow(rw)
#axarr[0, 1].imshow(image, cmap="gray")
#axarr[1, 0].imshow(markers)
#axarr[1, 1].imshow(watershed, cmap="gray")
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

def plotImages(rw, image):
    plt.imshow(rw)
    plt.show()
    plt.imshow(image, cmap="gray")
    plt.show()
    
def getSphere(rw):
    for x in np.nditer(rw, op_flags=['readwrite']):
        if x > 1:
            x[...] = 0
    return(rw)

def getPath(forset):
    outPath = "/home/adam/Optlev/SDP/results/" + forset + "/"
    return outPath
    
#other file    
    
def eliminateValues(values, lower, upper, forset):
    index = []
    for i in range(0, values.shape[0]):
        if (values[i] > upper or values[i] < lower):
            index = np.append(index, i)
    cutvalues = np.delete(values, index)      
    np.savetxt(getPath(forset) + str(lower) + str(upper) + ".txt", cutvalues, fmt="%1.8f")
    return(cutvalues)

def getPowers(values, eqpos, na, forset):
    powers = np.zeros(values.shape[0])
    for i in range(0, values.shape[0]):
        powers[i] = gp(values[i], eqpos, na)
    #np.savetxt(getPath(forset)+"spheres power data " + str(eqpos) + ".txt", powers, fmt="%1.8f")
    return(powers)

def combineFiles(num, forset):
    values = []
    for i in range(1, num+1):
        values = np.concatenate((values, np.loadtxt(getPath(forset) + str(i) + ".txt")))
    #np.savetxt(getPath(forset)+"full.txt", values, fmt="%1.8f")
    return values

def getArea(num, path):
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, str(num)+" *.txt"):
            return np.loadtxt(os.path.join(path,file))[0]
        
def combineAreas(num, path):
    values = []
    for i in range(1,num+1):
        values = np.append(values, getArea(i, path))
    return values

def getFullFile(num, path):
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, str(num)+" *s.txt"):
            return np.loadtxt(os.path.join(path, file))
            
def combineFullFiles(num, path):
    values = []
    for i in range(1, num+1):
        values = np.concatenate((values, getFullFile(i, path)))
    return values
            
def combineDataOld(num, path):
    values = []
    for i in range(1, num+1):
        values = np.append(values, np.loadtxt(os.path.join(path, str(i)+".txt"))[0])
    return values

def combineDiamsOld(num, path):
    values = []
    for i in range(1, num+1):
        image = io.imread(os.path.join(path, str(i)+".png"))
        colsums = np.sum(image, axis=0)
        values = np.append(values, np.amax(colsums)/(255*scale))
    return values
        
def getDiam(num, path):
    for file in os.listdir(path):
            if fnmatch.fnmatch(file, str(num)+" *0.png"):
                image = io.imread(os.path.join(path, file))
                colsums = np.sum(image, axis=0)
                return np.amax(colsums)/(255*scale)
                
def combineDiams(num, path):
    diams = []
    for i in range(1, num+1):
        diams = np.append(diams, getDiam(i, path))
    return diams

def getKSradialscale(scale, theory, real):
    powertheory = getPowers(theory*scale, 50, .0093, "set 2")
    #print(str(np.median(powertheory)) + " - " + str(scale))
    d, p = scipy.stats.ks_2samp(powertheory, real)
    return d, p

def getKSheightscale(scale, theory, real):
    powertheory = getPowers(theory, 50*scale, .0093, "set 2")
    #print(str(np.median(powertheory)) + " - " + str(scale))
    d, p = scipy.stats.ks_2samp(powertheory, real)
    return d, p

def getKSNAscale(scale, theory, real):
    powertheory = getPowers(theory, 50, .0093*scale, "set 2")
    #print(str(np.median(powertheory)) + " - " + str(scale))
    d, p = scipy.stats.ks_2samp(powertheory, real)
    return d, p

def getKSbothscale(radscale, heightscale, theory, real):
    powertheory = getPowers(radscale*theory, heightscale*50, .0093, "set 2")
    #print(str(np.median(powertheory)) + " - " + str(scale))
    d, p = scipy.stats.ks_2samp(powertheory, real)
    return d, p

def getX2bothscale(radscale, heightscale, theory, real):
    powertheory = getPowers(radscale*theory, heightscale*50, .0093, "set 2")
    #print(str(np.median(powertheory)) + " - " + str(scale))
    d, p = scipy.stats.chisquare(real, powertheory)
    return d, p

def getValues(forset, radtuple, heighttuple):
    radialscalevals = np.linspace(radtuple[0], radtuple[1], radtuple[2])
    heightscalevals = np.linspace(heighttuple[0], heighttuple[1], heighttuple[2])
    values = np.loadtxt(getPath(forset)+"full.txt")
    num = eliminateValues(values, 8, 25, forset)
    DC = [-540, -620, 20, 80, -620, 420, -670, -620, -25, -440, -510, -840, -180, -90, -500, -660, -340, 70, -500, -730, -400, -100, -285, -230, -510, -880, -90, -170, -600, -40, -240, -570, -110, -400, -250, -750, -760, -130, -590, -540, -400, -570, -235, -710, -640, -590, -510]
    power = 7.36428571e-02*np.array(DC) + 1.02128571e+02
    #print(np.median(power))
    #ds = []
    ps = []
    xs = []
    ys = []
    for i in range(0,radtuple[2]):
        for j in range(0,heighttuple[2]):
            dtemp, ptemp = getX2bothscale(radialscalevals[i], heightscalevals[j], num, power)
            #ds = np.append(ds, dtemp)
            ps = np.append(ps, ptemp)
            xs = np.append(xs, radialscalevals[i])
            ys = np.append(ys, heightscalevals[j])
    return xs, ys, ps

