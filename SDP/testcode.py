import skimage
from scipy import ndimage
from skimage import io
from skimage import segmentation
from skimage.feature import canny
from matplotlib import pyplot as plt
from skimage import morphology
import numpy as np
image = io.imread("C:/Users/JAA/Box Sync/Moore Lab Work/Sphere diameter project/good 2 seeded partial.png")
#thresh = skimage.filters.threshold_otsu(image)
#binary = image > thresh
#elevation = skimage.filters.sobel(image)
plt.imshow(image, cmap="gray")
markerdata = plt.ginput(10, timeout=-1)
np.save("C:/Users/JAA/Box Sync/Moore Lab Work/Sphere diameter project/data", markerdata)
x, y = np.load("C:/Users/JAA/Box Sync/Moore Lab Work/Sphere diameter project/data").T
plt.imshow(image, cmap='gray')
plt.plot(y, x, 'or', ms=4)
plt.show()
markers = np.zeros(image.shape, dtype=np.int)
print("made it here")
markers[x.astype(np.int), y.astype(np.int)] = np.arange(len(x)) + 1
markers = morphology.dilation(markers, morphology.disk(7))
#rw = segmentation.random_walker(image, markers, beta=130)
#print(watershed)
#watershed = skimage.morphology.watershed(image, markers)
#edge = canny(image, sigma=1.0)
#filled = ndimage.binary_fill_holes(image)
#plt.imshow(image, cmap="gray")
f, axarr = plt.subplots(2, 2)
#axarr[0, 0].imshow(rw)
axarr[0, 1].imshow(image, cmap="gray")
axarr[1, 0].imshow(markers)
#axarr[1, 1].imshow(watershed, cmap="gray")
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
#plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
#plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
#plt.imshow(watershed)
#plt.show()
#plt.imshow(markers)
plt.show()