from birefringence_grids import jones_calculation as jones_calculation_single
from birefringence_grids import get_chi_squared
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

path = "12-3-19_vaterite/"

npzfile = np.load(path + 'image_stack.npz')
array_of_images = npzfile["image_stack"]
alpha = npzfile["alpha_radians"]  # radians
beta = npzfile["beta_radians"]  # radians
# starting_parameters = npzfile["starting_parameters"]
n = len(alpha)
nj, nk = (array_of_images[0]).shape

start_bounds = [0., -np.pi, -np.pi, -np.pi, -np.pi, -np.pi]
end_bounds = [255., np.pi, np.pi, np.pi, np.pi, np.pi]


npzfile = np.load(path + 'original_grids_with_a114b112.npz')
starting_parameters = npzfile['new_parameters']


def jones_calculation(ab, c, phi, eta, theta, alpha_0, beta_0):
    # this is the matrix math result that I have to numerically solve for.
    a, b = zip(*ab)
    return c * jones_calculation_single(a, b, phi, eta, theta, alpha_0, beta_0)


def get_jones_parameters(intensity_measured, alpha_radians, beta_radians, p0):
    """Fits measured intensity data to the Jones calculation
       phi = circularity (0 for linear, +/- pi/2 for circular
       eta = relative phase retardation
       theta = orientation of fast axis w.r.t. x-axis (pi/4 for circular)"""
    popt, pcov = curve_fit(jones_calculation, zip(alpha_radians, beta_radians), intensity_measured,
                           bounds=(start_bounds, end_bounds), p0=p0)
    calculation = jones_calculation(zip(alpha_radians, beta_radians), *popt)
    chi_squared = get_chi_squared(intensity_measured - calculation)
    return popt, chi_squared


def jones_calculation_next(ab_a0b0, c, phi, eta, theta):
    # this is the matrix math result that I have to numerically solve for.
    a, b, a0, b0 = zip(*ab_a0b0)
    # print 'in jones_calculus_next'
    # print np.shape(a)
    # print np.shape(b)
    # print np.shape(a0)
    # print np.shape(b0)
    ans = c * jones_calculation_single(a, b, phi, eta, theta, a0, b0)
    # print np.shape(c), np.shape(phi), np.shape(eta), np.shape(theta)
    # print np.shape(ans)
    return ans


def get_jones_parameters_next(intensity_measured, alpha_radians, beta_radians, p0):
    """Fits measured intensity data to the Jones calculation
       phi = circularity (0 for linear, +/- pi/2 for circular
       eta = relative phase retardation
       theta = orientation of fast axis w.r.t. x-axis (pi/4 for circular)"""
    a0 = p0[-2] * np.ones(len(alpha_radians))
    b0 = p0[-1] * np.ones(len(beta_radians))
    # print alpha_radians.shape, beta_radians.shape, a0.shape, b0.shape
    # print np.shape(zip(alpha_radians, beta_radians, a0, b0))
    # print intensity_measured.shape
    popt, pcov = curve_fit(jones_calculation_next, zip(alpha_radians, beta_radians, a0, b0), intensity_measured,
                           bounds=(start_bounds[:-2], end_bounds[:-2]), p0=p0[:-2])
    calculation = jones_calculation_next(zip(alpha_radians, beta_radians, a0, b0), *popt)
    chi_squared = get_chi_squared(intensity_measured - calculation)
    return popt, chi_squared


def plot_pixel_fit(j, k, data, fit, guess):
    for_name = 'for ' + str(j) + ',' + str(k)

    vmin = np.min(zip(data, fit, guess))
    vmax = np.max(zip(data, fit, guess))
    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(fit, vmin=vmin, vmax=vmax)
    plt.title('fit ' + for_name)
    plt.subplot(132)
    plt.imshow(data, vmin=vmin, vmax=vmax)
    plt.title('data ' + for_name)
    plt.subplot(133)
    plt.imshow(guess, vmin=vmin, vmax=vmax)
    plt.title('guess for ' + for_name)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([1.05, 0.2, 0.03, 0.6])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.show()

    vmin = np.min(zip(data - fit, data - guess))
    vmax = np.max(zip(data - fit, data - guess))
    plt.figure(figsize=(7, 5))
    plt.subplot(121)
    plt.imshow(data - fit, vmin=vmin, vmax=vmax)
    plt.title('data - fit ' + for_name)
    plt.subplot(122)
    plt.imshow(data - guess, vmin=vmin, vmax=vmax)
    plt.title('data - guess ' + for_name)
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([1.05, 0.2, 0.03, 0.6])
    plt.colorbar(cax=cax)
    plt.tight_layout()
    plt.show()


def show_pixel_fit(j, k, initial_parameters, plot=True):
    """ this is from the original fit """
    intensity_list = array_of_images[:, j, k]
    # print intensity_list.shape
    # print alpha.shape
    # print beta.shape

    a0, b0 = initial_parameters[-2:]
    if a0 == b0 == 0.:
        popt, r = get_jones_parameters(intensity_list, alpha, beta, initial_parameters)
        print "popt is ", popt
        c, phi, eta, theta, a0, b0 = popt
    else:
        popt, r = get_jones_parameters_next(intensity_list, alpha, beta, initial_parameters)
        print "popt is ", popt
        c, phi, eta, theta = popt
    print 'chi^2 is ' + str(r)

    fit = np.zeros([10, 10])
    data = np.zeros([10, 10])
    guess = np.zeros([10, 10])
    for intensity, a_r, b_r in zip(intensity_list, alpha, beta):
        a = int(a_r * 9. / np.pi)
        b = - int(b_r * 9. / np.pi)
        fit[a, b] = jones_calculation_single(a_r, b_r, phi, eta, theta, a0, b0)
        data[a, b] = intensity
        guess[a, b] = jones_calculation_single(a_r, b_r, *tuple(initial_parameters[1:]))
    fit = c * fit
    guess = c * guess
    print 'data - fit has max ' + str(np.max(data-fit)) + ' and min ' + str(np.min(data-fit))
    print '    with median ' + str(np.median(data-fit))
    print 'guess - fit has max ' + str(np.max(guess-fit)) + ' and min ' + str(np.min(guess-fit))
    print '    with median ' + str(np.median(guess-fit))
    if plot: plot_pixel_fit(j, k, data, fit, guess)


def get_popt(intensity_list, j, k):
    popt = starting_parameters
    r = 0.
    try:
        popt, r = get_jones_parameters(intensity_list, alpha, beta, starting_parameters)
    except:
        print ('    curve_fit for j = ' + str(j) + ' and k = ' + str(k) + ' has failed')
    return popt, r, (j, k)


def get_original_grids(path):
    print ('    ')
    print ('beginning construction of the original grids')

    c_grid, phi_grid, eta_grid, theta_grid, a0_grid, b0_grid, r_grid = [np.zeros([nj, nk]) for i in range(7)]

    for k in range(nk):
        popts = Parallel(n_jobs=32)(delayed(get_popt)(array_of_images[:, j, k], j, k) for j in range(nj))
        for (c, phi, eta, theta, a0, b0), r, (j, _) in popts:
            c_grid[j, k] = c
            phi_grid[j, k] = phi
            eta_grid[j, k] = eta
            theta_grid[j, k] = theta
            r_grid[j, k] = r
            a0_grid[j, k] = a0
            b0_grid[j, k] = b0
        print "    "
        print "    done with column " + str(k) + " out of " + str(nk) + " columns"
        print "    "
    a0 = np.median(a0_grid)
    b0 = np.median(b0_grid)
    c0 = np.median(c_grid)
    p0 = np.median(phi_grid)
    e0 = np.median(eta_grid)
    t0 = np.median(theta_grid)
    a0_degrees = (a0 * 180. / np.pi).astype(int)
    b0_degrees = (b0 * 180. / np.pi).astype(int)
    new_parameters = (c0, p0, e0, t0, a0, b0)
    file_name = path + 'original_grids_with_a' + str(a0_degrees) + 'b' + str(b0_degrees) + '.npz'
    np.savez(file_name, c=c_grid, phi=phi_grid, eta=eta_grid, theta=theta_grid, residuals=r_grid,
             a0=a0_grid, b0=b0_grid, new_parameters=new_parameters)
    return new_parameters


def get_new_popt(intensity_list, j, k):
    popt = starting_parameters[:-2]
    r = 0.
    try:
        popt, r = get_jones_parameters_next(intensity_list, alpha, beta, starting_parameters)
    except:
        print ('    curve_fit for j = ' + str(j) + ' and k = ' + str(k) + ' has failed')
    return popt, r, (j, k)
    

def get_renormalized_grids():
    print ('    ')
    print ('    beginning construction of the new grids')
    c_grid, phi_grid, eta_grid, theta_grid, r_grid = [np.zeros([nj, nk]) for i in range(5)]
    for k in range(nk):
        popts = Parallel(n_jobs=32)(delayed(get_new_popt)(array_of_images[:, j, k], j, k) for j in range(nj))
        for (c, phi, eta, theta), r, (j, _) in popts:
            c_grid[j, k] = c
            phi_grid[j, k] = phi
            eta_grid[j, k] = eta
            theta_grid[j, k] = theta
            r_grid[j, k] = r
        print "    "
        print "    done with column " + str(k) + " out of " + str(nk) + " columns"
        print "    "

    a0_degrees = int(starting_parameters[-2] * 180 / np.pi)
    b0_degrees = int(starting_parameters[-1] * 180 / np.pi)
    file_name = path + 'renormalized_grids_with_a' + str(a0_degrees) + 'b' + str(b0_degrees) + '.npz'
    np.savez(file_name, c=c_grid, phi=phi_grid, eta=eta_grid, theta=theta_grid, residuals=r_grid)


# new_starting_parameters = get_original_grids(folder_name)
get_renormalized_grids()
