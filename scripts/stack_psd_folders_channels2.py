# more efficient stack_psd_folders_channels.py

import h5py, glob, os
from matplotlib.mlab import psd as get_psd
import matplotlib.pyplot as plt
import numpy as np
import bead_util as bu

read_signal_npy = True

# plot_in_plot_V = [True]

folder_signal = [r"C:\data\20180806\electronic_noise_trek_4kVpp_AI1ondaq_laser_on"]

folder_list0 = [r"C:\data\20180817\transfer_function_before_chamber_X",
               r"C:\data\20180817\transfer_function_before_chamber_Y",
               r"C:\data\20180817\transfer_function_before_chamber_0",
               r"C:\data\20180816\transfer_function_0_gain_after_chamber",
               r"C:\data\20180816\transfer_function_y_gain_0.01_after_chamber",
               r"C:\data\20180816\transfer_function_x_gain_0.01_after_chamber"]

folder_list1 = [r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber",
                # r"C:\data\20180816\laser_on_5kV_trek_on_battery_unplugged_on_roof_1064nm_after_chamber",
                r"C:\data\20180816\laser_on_5kV_trek_off_on_roof_1064nm_after_chamber",
                # r"C:\data\20180816\laser_on_5kV_trek_on_high_voltage_off_on_roof_1064nm_after_chamber",
                r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber_2",
                r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber_y_unplugged",]

folder_list2 = [r"C:\data\20180817\trek_5kV_opposite_corner_box_on_roof_after_chamber",
                # r"C:\data\20180817\trek_20kV_opposite_corner_box_on_roof_before_chamber",
                # r"C:\data\20180820\trek_20kV_opposite_corner_of_room_before_chamber_laser_on",
                r"C:\data\20180820\trek_20kV_opposite_corner_of_room_after_chamber_laser_on",
                r"C:\data\20180820\trek_20kV_opposite_corner_on_roof_after_chamber_laser_on",
                r"C:\data\20180820\trek_20kV_opposite_corner_on_roof_gloves_after_chamber_laser_on"]

folder_list3 = [r"C:\data\20180831\trek_20kV_opposite_corner_of_room_532X_1064Y_after_chamber",
                r"C:\data\20180831\trek_off_opposite_corner_of_room_532X_1064Y_after_chamber"]

folderlist35 = [r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber_2"]

folder_list4 = [r"C:\data\20180911\20kV_X532_Y1064_connected_to_chamber_opposite_corner_of_room",
                r"C:\data\20180911\20kV_X532_Y1064_not_connected_to_chamber_opposite_corner_of_room"]

folder_list5 = [r"C:\data\20180702\bead2_SiO2_15um_POL_NS\charge"]


folder_list = folder_list5#35 + folder_list4

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 19


def getdata(fname, drive):
    print "Opening file: ", fname
    ## guess at file type from extension
    _, fext = os.path.splitext(fname)
    if (fext == ".h5"):
        f = h5py.File(fname, 'r')
        dset = f['beads/data/pos_data']
        dat = np.transpose(dset)
        max_volt = 10.
        nbit = 2 ** 15 - 1
        Fs = dset.attrs['Fsamp']
        dat = 1.0 * dat * max_volt / nbit
    else:
        dat = np.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

    if drive:
        dpsd, freqs = get_psd(dat[:, bu.drive] - np.mean(dat[:, bu.drive]), Fs=Fs, NFFT=NFFT)
        return freqs, dpsd, dpsd
    else:
        xpsd, freqs = get_psd(dat[:, bu.xi] - np.mean(dat[:, bu.xi]), Fs=Fs, NFFT=NFFT)
        ypsd, freqs = get_psd(dat[:, bu.yi] - np.mean(dat[:, bu.yi]), Fs=Fs, NFFT=NFFT)
        return freqs, xpsd, ypsd


def stack_psd(folder, drive):
    if drive:
        file_list = glob.glob(folder + "\*.h5")
        if len(folder) == 0:
            print "error empty folder"
            print folder
            return

        freq, p2, p2 = getdata(file_list[0], drive)
        p2 = np.zeros(len(p2))
        for i in file_list:
            f, aux, aux = getdata(i, drive)
            p2 = p2 + aux
        p2 = p2 / len(file_list)
        return freq, p2, p2
    else:
        file_list = bu.time_ordered_h5_and_npy_file_list(folder, False)
        n = len(file_list)
        if n == 0: 
            print "error empty folder"
            print folder
            return

        try:
            x_name = os.path.join(folder, "measurement_x.npy")
            xind = file_list.index(x_name)
            y_name = os.path.join(folder, "measurement_y.npy")
            yind = file_list.index(y_name)
            xmeas = np.load(x_name)
            ymeas = np.load(y_name)
            freq, xp2 = xmeas
            freq, yp2 = ymeas
            index = max(xind, yind)
            if index == n:
                return freq, xp2, yp2
            else:
                n = n - 2
                xp2 = xp2 * float(index - 2)
                yp2 = yp2 * float(index - 2)
        except:
            print "making new npy"
            freq, xp2, yp2 = getdata(file_list[0], drive)
            index = 0

        for i in file_list[index + 1:]:
            f, xaux, yaux = getdata(i, drive)
            xp2 = xp2 + xaux
            yp2 = yp2 + yaux

        xp2 = xp2 / float(n)
        yp2 = yp2 / float(n)

        np.save(os.path.join(folder, "measurement_x"), [freq, xp2])
        np.save(os.path.join(folder, "measurement_y"), [freq, yp2])
        return freq, xp2, yp2


def read_psd_folders(folder_list, drive):
    """ returns lists """
    freqs = []
    xpsds = []
    ypsds = []
    names = []
    for i in range(len(folder_list)):
        freq, xpsd2, ypsd2 = stack_psd(folder_list[i], drive)
        freqs.append(freq)
        xpsds.append(xpsd2)
        ypsds.append(ypsd2)
        names.append(str(folder_list[i]))
    return freqs, xpsds, ypsds, names


def peak_position(folder):
    if read_signal_npy:
        try:
            np.load(os.path.join(folder[0], "position.npy"))  # the zero is because the way things are structured.
            return np.load(os.path.join(folder[0], "position.npy"))
        except:
            print "no npy"

    f, dp2, dp2, name = read_psd_folders(folder, True)
    a = np.argmax(dp2)
    np.save(os.path.join(folder[0], "position"), a)
    return a


def plot_single_axis(name, f, p2, arg, channel):
    area = []
    plt.figure()
    for i in range(len(f)):
        curr_label = name[i]
        curr_label = curr_label[curr_label.find('\\', 16) + 1:]
        plt.loglog(f[i], np.sqrt(p2[i]), label=curr_label)
        a = np.sum(p2[i][arg - 1:arg + 2])
        a = np.sqrt(a)
        area.append(a)
    plt.axvline(f[0][arg], color="r", linestyle="--")
    plt.legend()
    plt.xlim(48 - 2, 48 + 2)
    plt.ylim(1e-5, 1e-3)
    plt.xlabel("Frequency [Hz]")
    if channel == bu.xi:
        name_channel = " of channel X"
    elif channel == bu.yi:
        name_channel = " of channel Y"
    else:
        name_channel = ""
    plt.ylabel("PSD [V/$\sqrt{Hz}$]" + name_channel)
    plt.grid()
    plt.tight_layout(pad=0)
    return area


def plot_all(folder_list, folder_signal):
    arg = peak_position(folder_signal)
    f, xp2, yp2, name = read_psd_folders(folder_list, False)
    xarea = plot_single_axis(name, f, xp2, arg, bu.xi)
    yarea = plot_single_axis(name, f, yp2, arg, bu.yi)
    # name = ["original disconnected 5kV", "moved 20kV", "moved disconnected 20kV"]
    return xarea, yarea, name


def plot_XY_together(folder_list, folder_signal):
    arg = peak_position(folder_signal)
    f, xp2, yp2, name = read_psd_folders(folder_list, False)
    # names = ["532nm trek_20kV_connected", "532nm trek_20kV_not_connected", "1064nm trek_20kV_connected", "1064nm trek_20kV_not_connected"]
    # names = [i + "'s X input" for i in name] + [i + "'s Y input" for i in name]
    plot_single_axis(names, f + f, xp2 + yp2, arg, "")


def plot_V(Area, Name, X, channel):
    plt.figure()
    for i in range(len(X)):
        if X[i]:
            try:
                vpp = Name[i].split("kV")[0]
                vpp = vpp[vpp.rfind("_") + 1:]
                vpp = float(vpp)
            except:
                vpp = 11.0
            curr_label = Name[i]
            if 'farther_electrodes' in curr_label:
                new_folder = curr_label.split('farther_electrodes')[1]
                new_folder = new_folder.split('electronic_noise')[0]
                if new_folder == '\\':
                    symb = "*"
                else:
                    symb = "^"
            else:
                symb = "o"
            curr_label = curr_label[curr_label.find('\\', 16) + 1:]
            plt.plot(vpp, Area[i], symb, label=curr_label)
    plt.xlabel("Vpp_daq [V]")
    if channel == bu.xi:
        name_channel = "X"
    else:
        name_channel = "Y"
    plt.ylabel("Volts " + name_channel + " sensor (100 gain) [V]")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0)
    return


#x_area, y_area, name_list = 
plot_all(folder_list, folder_signal)

# plot_XY_together(folder_list, folder_signal)

# plot_V(x_area, name_list, plot_in_plot_V, bu.xi)
# plot_V(y_area, name_list, plot_in_plot_V, bu.yi)
plt.show()
