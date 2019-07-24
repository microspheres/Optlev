# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.cm as cmx
# import matplotlib.colors as colors

path_name = r'C:\Users\Sumita\Documents\Research\Microspheres\vaterite spheres\2019-05-09'

f = np.load(path_name + '\\diameters_and_roundness_data.npz')

diameters = f['diameters']
# roundness = f['roundness']
aspect_ratios = f['aspect_ratios']
sphere_labels = f['labels']
sphere_markers = f['markers']
# sphere_labels = ['Sample #' + str(i + 1) for i in range(6)]

save_figures = True

# now calculate the number of bins needed


def calculate_bins(data, starting_bins, desired_bins_per_sigma):
    # print data
    counts, bins = np.histogram(data, bins=starting_bins)
    # print counts
    mean = np.average(data)
    # print mean
    sigma = np.sqrt(np.average((data - mean) ** 2))
    # print sigma
    num_bins_per_sigma = sigma / float(bins[1] - bins[0])
    return int(starting_bins * desired_bins_per_sigma / num_bins_per_sigma)


diameters_bins = []
# roundness_bins = []
aspectrat_bins = []
for a, d in zip(aspect_ratios, diameters):
    diameters_bins.append(calculate_bins(d, 8, 3))
    # r = np.array(r)
    # roundness_bins.append(calculate_bins(r[r > 0.5], 8, 3))
    a = np.array(a)
    aspectrat_bins.append(calculate_bins(a[a < 1.5], 8, 3))
#
# num_bins_array = [40, 60, 30, 30, 30, 200]
#
#
# def gauss(x, mu, sigma, a):
#     return a * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)
#
#
# def half_gauss(x, sigma, a):
#     return gauss(x, 1, sigma, a)
#

# def get_fit(sphere_label, curr_x, curr_y, type):
#     """
#     gets the fit to a gaussian or half gaussian for a histogram
#     :param sphere_label: Name of the sphere
#     :param curr_x: values for the parameter given by 'type'
#     :param curr_y: number of spheres for a given value of that parameter
#     :param type: string, diameter, roundness, or aspect ratio
#     :return: long label for data and the fit on the gaussian
#     """
#     x_pts = np.linspace(np.min(curr_x), np.max(curr_x))
#     if type == 'diameter':
#         popt, pcov = curve_fit(gauss, curr_x, curr_y, bounds=[0, 50])
#         errorbar_label = sphere_label + ' with $\mu=$%.3f' % popt[0] + ' and $\sigma=$%.3f' % popt[1]
#         y_pts = gauss(x_pts, *popt)
#     else:
#         popt, pcov = curve_fit(half_gauss, curr_x, curr_y, bounds=[0, 5])
#         errorbar_label = sphere_label + ' with $\sigma=$%.3f' % popt[0]
#         y_pts = half_gauss(x_pts, *popt)
#     return errorbar_label, x_pts, y_pts


# def get_color_map(n):
#     jet = plt.get_cmap('jet')
#     c_norm = colors.Normalize(vmin=0, vmax=n - 1)
#     scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)
#     outmap = []
#     for i in range(n):
#         outmap.append(scalar_map.to_rgba(i))
#     return outmap
#
#
# colormap = get_color_map(len(diameters))


def save_histogram(data, num_bins_array, arb_heights, data_type, max_height, units=''):
    """
    Saves a histogram with error bars and everything
    :param data: data to be histogrammed (list of lists)
    :param num_bins_array: number of bins per set of data (int)
    :arb_heights: height of peak of plot (makes it prettier)
    :param data_type: diameter or roundness or aspect ratio (string)
    :param units: string
    :return: plot
    """

    plt.figure(figsize=(5, 4))
    for i, (d, b, h, m, l) in enumerate(zip(data, num_bins_array, arb_heights, sphere_markers, sphere_labels)):
        if data_type == 'aspect ratio':
            d = np.array(d)
            d = d[d < 1.5]
        elif data_type == 'roundness':
            d = np.array(d)
            d = d[d > 0.5]
        print l
        print np.average(d)
        print np.sqrt(np.average((np.average(d) - d) ** 2))
        y, bins = np.histogram(d, bins=b)
        x = (bins[1:] + bins[:-1]) / 2
        # x = bins[:-1]  # for this, set where='post'
        # ebl, x_fit, y_fit = get_fit(sl, x, y / float(max(y)))
        # yerr = np.sqrt(y) / float(max(y))
        # plt.errorbar(x, y / float(max(y)), fmt=fmt, color=c, yerr=yerr, fillstyle='none', label=ebl)
        y_plot = 100. * y / float(len(d) * max_height)  # float(max(y)) * h
        plt.step(x, y_plot, where='mid', color='C' + str(i), label='Sample ' + str(i + 1))
    #     max_ind = np.argmax(y_plot)
    #     plt.plot([x[max_ind]], [y_plot[max_ind]], m, color='C' + str(i), fillstyle='none')
    #     plt.plot(x, y_plot, m, color='C' + str(i), fillstyle='none')
    #     plt.plot(x_fit, y_fit, color=c)
    # plt.title('Histogram of sphere ' + data_type)
    if units == '':
        plt.xlabel('Sphere ' + data_type)
    else:
        plt.xlabel('Sphere ' + data_type + ' [' + units + ']')
    plt.ylabel('Distribution [arb]')
    if data_type == 'aspect ratio':
        plt.xlim((1.0, 1.5))
    elif data_type == 'roundness':
        plt.xlim((0.5, 1.0))
    plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    if save_figures:
        plt.savefig(path_name + '\\' + data_type + '_histogram_normed.png')  # , bbox_inches="tight")
    plt.show()


# plotting diameter histogram
# heights = [1, 0.85, 1.1, 1, 0.6, 0.6]
heights = np.ones(6)
save_histogram(diameters, diameters_bins, heights, 'diameter', 30, units='um')
#
# # now plotting roundness histogram
# save_histogram(roundness, roundness_bins, 'roundness')

# finally plotting aspect ratio histogram
# heights = [0.8, 0.3, 1.1, 1.15, 1.05, 1.1]
# heights = np.array([60, 40, 20, 25, 10, 60]) / 60.
save_histogram(aspect_ratios, aspectrat_bins, heights, 'aspect ratio', 25)
