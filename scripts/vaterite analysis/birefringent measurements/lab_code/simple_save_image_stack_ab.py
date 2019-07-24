import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from os.path import join
from scipy.misc import imread

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"
extension = ".png"

path_name = default_path + '12-3-19_vaterite\\'
file_list = glob(join(path_name, '*0b*' + extension))


def read_name_degrees(file_name):
    """reads off angle(s) from file name
       change to add beta once applicable"""
    i = file_name.rfind('a') + 1
    j = file_name.find('.')
    k = file_name.find('b', i)
    alpha_in_degrees = int(file_name[i:k])
    beta_in_degrees = int(file_name[k + 1:j])
    return alpha_in_degrees, beta_in_degrees


im0 = imread(file_list[0], flatten=True)
nj, nk = im0.shape

grid_of_images = np.zeros([9, 9, nj, nk])
image_stack = []  # original image
alpha_degrees = []  # in degrees
beta_degrees = []  # in degrees
background_brightness = np.zeros([9, 9])
for f in file_list:
    image = imread(f, flatten=True)
    a, b = read_name_degrees(f)
    background_brightness[a / 20, b / 20] = np.median(image)
    grid_of_images[a/20, b/20] = image
    image_stack.append(image)
    alpha_degrees.append(a)
    beta_degrees.append(b)


# plt.figure()
# plt.imshow(background_brightness)
# plt.show()

c_guess = float(background_brightness.max())
a, b = np.unravel_index(background_brightness.argmax(), background_brightness.shape)
background_brightness = np.roll(background_brightness, -a, axis=0)
background_brightness = np.roll(background_brightness, -b, axis=1)
grid_of_images = np.roll(grid_of_images, -a, axis=0)
grid_of_images = np.roll(grid_of_images, -b, axis=1)
a0 = a * 20
b0 = b * 20
alpha_degrees = np.array(alpha_degrees) - a0
beta_degrees = np.array(beta_degrees) - b0
alpha_radians = np.array(alpha_degrees) * np.pi / 180.
beta_radians = np.array(beta_degrees) * np.pi / 180.

a0_guess = a0 * np.pi / 180.
b0_guess = b0 * np.pi / 180.
phi_guess, eta_guess, theta_guess = [0. for i in range(3)]
starting_parameters = [c_guess, phi_guess, eta_guess, theta_guess, a0_guess, b0_guess]

np.savez(path_name + 'image_stack.npz', image_stack=image_stack, alpha_radians=alpha_radians, beta_radians=beta_radians,
         alpha_degrees=alpha_degrees, beta_degrees=beta_degrees, background_brightness=background_brightness,
         starting_parameters=starting_parameters, grid_of_images=grid_of_images)
