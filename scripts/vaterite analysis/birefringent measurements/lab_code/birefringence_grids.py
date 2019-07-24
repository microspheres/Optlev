from scipy import misc
# from glob import glob
# from os.path import join
import numpy as np
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

default_path = "C:\\Users\\Sumita\\Documents\\Research\\Microspheres\\birefringent measurements\\sphere images\\"


def read_name_degrees(file_name):
    """reads off angle(s) from file name
       change to add beta once applicable"""
    i = file_name.rfind('a') + 1
    j = file_name.find('.', i)
    k = file_name.find('b', i)
    alpha_in_degrees = int(file_name[i:k])
    beta_in_degrees = int(file_name[k + 1:j])
    return alpha_in_degrees, beta_in_degrees


def show_image(f, vmin = 0, vmax = 0, title='', block=True):
    # .astype(np.uint8)
    if type(f) is str:
        f = misc.imread(f, flatten=True)
    plt.figure()
    if vmin == vmax:
        plt.imshow(f)
    else:
        plt.imshow(f, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.show(block=block)


def jones_matrix_entries(phi, eta, theta):
    arb_11 = np.exp(1j * eta / 2.) * (np.cos(theta) ** 2) + np.exp(-1j * eta / 2.) * (np.sin(theta) ** 2)
    arb_12 = 1j * np.exp(-1j * phi) * np.sin(eta / 2.) * np.sin(2 * theta)
    arb_21 = 1j * np.exp(1j * phi) * np.sin(eta / 2.) * np.sin(2 * theta)
    arb_22 = np.exp(-1j * eta / 2.) * (np.cos(theta) ** 2) + np.exp(1j * eta / 2.) * (np.sin(theta) ** 2)
    return arb_11, arb_12, arb_21, arb_22


def linear_polarizer(theta):
    lph_11 = np.cos(theta) ** 2
    lph_12 = np.sin(theta) * np.cos(theta)
    lph_21 = np.sin(theta) * np.cos(theta)
    lph_22 = np.sin(theta) ** 2
    return lph_11, lph_12, lph_21, lph_22


def full_intensity(alpha_radians, beta_radians, phi, eta, theta):
    """ 
    :param alpha_radians: angle of polarization of incoming light
    :param beta_radians: angle of polarizer for outgoing light
    :param phi: circularity
    :param eta: birefringence
    :param theta: angle to the fast-axis
    :return: intensity seen by the camera
    """
    initial_x = np.cos(alpha_radians)
    initial_y = np.sin(alpha_radians)
    arb_11, arb_12, arb_21, arb_22 = jones_matrix_entries(phi, eta, theta)
    lph_11, lph_12, lph_21, lph_22 = linear_polarizer(beta_radians)
    upon_escape_x = initial_x * arb_11 + initial_y * arb_12
    upon_escape_y = initial_x * arb_21 + initial_y * arb_22
    into_camera_x = upon_escape_x * lph_11 + upon_escape_y * lph_12
    into_camera_y = upon_escape_x * lph_21 + upon_escape_y * lph_22
    intensity_measured = np.abs(into_camera_x) ** 2 + np.abs(into_camera_y) ** 2
    return intensity_measured


def jones_calculation(a, b, phi, eta, theta, alpha_0, beta_0):
    # this is the matrix math result that I have to numerically solve for.
    alpha = np.array(a) + np.array(alpha_0)
    beta = np.array(b) + np.array(beta_0)
    # x1 = np.cos(beta) * np.cos(alpha) * (
    #     np.exp(1j * eta / 2) * (np.cos(theta)) ** 2 + np.exp(-1j * eta / 2) * (np.sin(theta)) ** 2)
    # x2 = np.sin(beta) * np.sin(alpha) * (
    #     np.exp(-1j * eta / 2) * (np.cos(theta)) ** 2 + np.exp(1j * eta / 2) * (np.sin(theta)) ** 2)
    # x3 = np.cos(beta) * np.sin(alpha) * np.sin(eta / 2) * np.sin(2 * theta) * np.exp(-1j * phi)
    # x4 = np.sin(beta) * np.cos(alpha) * np.sin(eta / 2) * np.sin(2 * theta) * np.exp(1j * phi)
    # x = x1 + x2 + x3 + x4
    # return np.abs(x) ** 2
    return full_intensity(alpha, beta, phi, eta, theta)


def get_chi_squared(differences):
    """
    :param differences: list of intensity - fit
    :return: chi^2 value
    """
    diff_list = np.ravel(differences) ** 2
    var = np.sum(diff_list)
    data_error_squared = np.median(diff_list)
    chi_squared = var / data_error_squared
    return chi_squared


def get_residuals(alpha_list, beta_list, intensity_list, phi, eta, theta, alpha_0, beta_0):
    """ this is what can change as needed """
    differences = np.zeros(len(intensity_list))
    for index, (a, b, i) in enumerate(zip(alpha_list, beta_list, intensity_list)):
        fit = jones_calculation(a, b, phi, eta, theta, alpha_0, beta_0)
        differences[index] = i - fit
    return get_chi_squared(differences)


def get_residuals_array(grid_npzfile, image_stack_npzfile, a0, b0):
    """ assumes that divide_by = 1 and that this is the new one """
    array_of_images = image_stack_npzfile['image_stack']
    a = image_stack_npzfile['alpha_radians']
    b = image_stack_npzfile['beta_radians']
    c = grid_npzfile['c']
    phi = grid_npzfile['phi']
    eta = grid_npzfile['eta']
    theta = grid_npzfile['theta']
    nj, nk = np.shape(c)
    r_grid = np.zeros([nj, nk])
    print 'getting the residuals array'
    print 'going on grids of ' + str(nj) + ' by ' + str(nk)
    for j in range(nj):
        for k in range(nk):
            intensity_list = array_of_images[:,j,k]
            r_grid[j, k] = get_residuals(a, b, intensity_list, phi[j, k], eta[j, k], theta[j, k], a0, b0)
        print 'done with row ' + str(j)
    return r_grid


def replace_residuals_array(npzfile_name, im_npzfile_name):
    """ assumes this is the file with fixed a0 and b0 """
    a0, b0 = read_name_degrees(npzfile_name)
    npzfile = np.load(npzfile_name)
    im_npzfile = np.load(im_npzfile_name)
    c_grid = npzfile['c']
    phi_grid = npzfile['phi']
    eta_grid = npzfile['eta']
    theta_grid = npzfile['theta']
    r_grid = get_residuals_array(npzfile, im_npzfile, a0, b0)
    np.savez(npzfile_name, c=c_grid, phi=phi_grid, eta=eta_grid, theta=theta_grid, residuals=r_grid)


def save_variable_grid(grid, grid_name, path_name, directory_name, block=False):
    plt.figure()
    plt.imshow(grid)
    plt.title(grid_name + ' of ' + directory_name)
    plt.colorbar()
    plt.savefig(path_name + grid_name + '.png')
    plt.show(block=block)


def save_grids(file_name, directory_name):
    """ currently saves only eta and sin/cos eta """
    path_name = default_path + directory_name + "\\aligned\\grids\\"
    file_path_name = path_name + file_name + ".npz"
    npzfile = np.load(file_path_name)
    c_grid = npzfile['c']
    eta_grid = npzfile['eta']
    r_grid = npzfile['residuals']  # this is np.average(image - calculation) / coefficient

    names = ('coefficient', 'eta', 'cos_eta', 'sin_eta', 'r_divided_by_c')
    if file_name == "grids_old.npz":
        old_names = [n + '_old' for n in names]
        names = old_names

        save_variable_grid(npzfile['a0'], "a0", path_name, directory_name)
        save_variable_grid(npzfile['b0'], "b0", path_name, directory_name)
    save_variable_grid(c_grid, names[0], path_name, directory_name)
    save_variable_grid(eta_grid, names[1], path_name, directory_name)
    save_variable_grid(np.cos(eta_grid), names[2], path_name, directory_name)
    save_variable_grid(np.sin(eta_grid), names[3], path_name, directory_name)
    save_variable_grid(r_grid, names[4], path_name, directory_name, block=True)
    return

# dir_name = "glass 10X 3-9-18"
# npz_name = "grids_with_a-128b146"
# save_grids(npz_name, dir_name)
