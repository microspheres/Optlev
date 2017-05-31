import skimage
from scipy import ndimage
from skimage import io
from matplotlib import pyplot as plt
import numpy as np

def getPoints(num):
	image = io.imread("/home/adam/Sphere diameter project/images/fernando/" + str(num) + ".png")
	plt.imshow(image, cmap="gray")
	markerdata = plt.ginput(n=-1, timeout=-1)
	np.save("/home/adam/Sphere diameter project/data/fernando/" + str(num), markerdata)

for i in range(23,33):
	getPoints(i)