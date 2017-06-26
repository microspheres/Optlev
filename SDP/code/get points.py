import skimage
from scipy import ndimage
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

def getPoints(num):
	image = io.imread("/home/adam/Optlev/SDP/images/microscope/" + str(num) + ".png")
	plt.imshow(image, cmap="gray")
	markerdata = plt.ginput(n=-1, timeout=-1)
	np.save("/home/adam/Optlev/SDP/data/microscope/" + str(num), markerdata)

for i in range(1,43):
	getPoints(i)