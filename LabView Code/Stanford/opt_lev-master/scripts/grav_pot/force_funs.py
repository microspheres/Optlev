import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt_grid = True
plt_b_grid = True

fr = lambda r: np.exp(-1.*r)/r

x = np.linspace(.001, 3, num = 100)
y = fr(x)

#generate x y and z coordinates for each point in the attractor

xmin = -1.
xmax = 1.
ymin = -1.
ymax = 1.
zmin = 0.
zmax = 10.
a  = .1
npts = int(np.round((xmax-xmin)*(ymax-ymin)*(zmax-zmin)/a**3))
att_grid = np.reshape(np.mgrid[xmin:xmax:a, ymin:ymax:a, zmin:zmax:a].T, (npts, 3))
xs = att_grid[:, 0]
ys = att_grid[:, 1]
zs = att_grid[:, 2]


if plt_grid:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    #plt.show()

#generate y, y , and z coordinates where the bead going to be

b_xmin = 0.
b_xmax = 10.
ax = .1

b_ymin = 0.
b_ymax = 1.
ay = 1.

b_zmin = -2.
b_zmax = -1.
az= 1.

b_npts = int(np.round((b_xmax-b_xmin)*(b_ymax-b_ymin)*(b_zmax-b_zmin)/(ax*ay*az)))
b_grid = np.reshape(np.mgrid[b_xmin:b_xmax:ax, b_ymin:b_ymax:ay, b_zmin:b_zmax:az].T, (b_npts, 3))
bxs = b_grid[:, 0]
bys = b_grid[:, 1]
bzs = b_grid[:, 2]


if plt_b_grid:
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax.scatter(bxs, bys, bzs)
    plt.show()
