import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit

# read_npy = [0, 0, 0, 0, 0, 0, 0, 0, 0] # to rewrite all npy files
read_npy = [1, 1] # 1 means do NOT read h5
read_signal_npy = True

Area_data = [1, 1, 1, 1, 1] # 1 means plot
Area_data2 = [1, 1] # 1 means plot

folder_signal = [r"C:\data\20180806\electronic_noise_trek_4kVpp_AI1ondaq_laser_on"]

folder_list1 = [r"C:\data\20180806\electronic_noise_trek_4kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_0.4kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_2kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_1kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_3kVpp_AI1offdaq_laser_on",]

folder_list2 = [r"C:\data\20180806\farther_electrodes\electronic_noise_trek_3kVpp_AI1offdaq_laser_on",
               r"C:\data\20180806\farther_electrodes\electronic_noise_trek_4kVpp_AI1offdaq_laser_on",
               r"C:\data\20180806\farther_electrodes\electronic_noise_trek_2kVpp_AI1offdaq_laser_on",
               r"C:\data\20180806\farther_electrodes\electronic_noise_trek_0.5kVpp_AI1offdaq_laser_on",
               r"C:\data\20180806\farther_electrodes\electronic_noise_trek_1kVpp_AI1offdaq_laser_on"]

folder_list3 = [r"C:\data\20180806\farther_electrodes\electronic_noise_trek_4kVpp_AI1offdaq_laser_off",]
                #r"C:\data\20180806\farther_electrodes\47Hz\electronic_noise_trek_4kVpp_AI1offdaq_laser_on"]

folder_list4 = [#r"C:\data\20180806\before_chamber\vertical_electronic_noise_trek_4kVpp_AI1offdaq_laser_on",
                #r"C:\data\20180806\before_chamber\electronic_noise_trek_4kVpp_AI1offdaq_laser_off",
                #r"C:\data\20180806\before_chamber\horizontal_electronic_noise_trek_off_AI1offdaq_laser_on",
                r"C:\data\20180810\TREK_unplugged\laser_on_trek_4kVpp_x",]

folder_list5 = [r"C:\data\20180810\laser_on_4kV_old_XY",
                r"C:\data\20180810\laser_off_4kV_old_XY"]

folder_list = folder_list3 + [folder_list2[1]]

folder_list_s = folder_list3 + folder_list4

channel = bu.xi

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2**19

def getdata(fname, drive):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['Fsamp']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2, 3, 4, 5, 6] )

        if drive:
            xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.drive]-numpy.mean(dat[:, bu.drive]), Fs = Fs, NFFT = NFFT)
        else:
            xpsd, freqs = matplotlib.mlab.psd(dat[:, channel]-numpy.mean(dat[:, channel]), Fs = Fs, NFFT = NFFT)
            
	return [freqs, xpsd]

def stack_psd(folder, drive, a):
    
    if a == 1 and drive == False:
        try:
            if channel == bu.xi:
                np.load(os.path.join(folder, "measurement_x.npy"))
                return np.load(os.path.join(folder, "measurement_x.npy"))
            else:
                np.load(os.path.join(folder, "measurement_y.npy"))
                return np.load(os.path.join(folder, "measurement_y.npy"))
        except:
            print "no npy"

    file_list = glob.glob(folder+"\*.h5")
    if len(folder) == 0:
        return "error empty folder"

    freq, p2 = getdata(file_list[0], drive)
    p2 = np.zeros(len(p2))
    for i in file_list:
        f, aux = getdata(i, drive)
        p2 = p2 + aux
    
    if not drive:
        if channel == bu.xi:
            np.save(os.path.join(folder, "measurement_x"), [freq, p2/len(file_list)])
        else:
            np.save(os.path.join(folder, "measurement_y"), [freq, p2/len(file_list)])
    
    return [freq, p2/len(file_list)]


def read_psd_folders(folder_list, drive):
    Psd2 = []
    Freq = []
    folder_name = []
    for i in range(len(folder_list)):
        freq, psd2 = stack_psd(folder_list[i], drive, read_npy[i])
        Freq.append(freq)
        Psd2.append(psd2)
        folder_name.append(str(folder_list[i]))

    return [Freq, Psd2, folder_name]


def peak_position(folder):
    if read_signal_npy:
        try:
            np.load(os.path.join(folder[0], "position.npy")) # the zero is because the way things are structured.
            return np.load(os.path.join(folder[0], "position.npy"))
        except:
            print "no npy"

    f, p2, name = read_psd_folders(folder, True)
    a = np.argmax(p2)
    np.save(os.path.join(folder[0], "position"), a)
    return a
    
    
def plot_all(folder_list, folder_signal):
    arg = 2*peak_position(folder_signal)
    F, P2, Name = read_psd_folders(folder_list, False)
    Area = []
    plt.figure()
    for i in range(len(folder_list)):
        curr_label = Name[i]
        curr_label = curr_label[curr_label.find('\\', 16)+1:]
        plt.loglog(F[i], np.sqrt(P2[i]), label = curr_label)
        a = np.sum(P2[i][arg-1:arg+1])
        a = np.sqrt(a)
        Area.append(a)
    plt.axvline(F[0][arg], color = "r", linestyle = "--")
    # plt.legend()
    plt.xlim(2*48-2, 2*48+2)
    plt.ylim(1e-5, 1e-2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V/$\sqrt{Hz}$]")
    plt.grid()
    
    return [Area, Name]

def func(x, a, b, c):
    y = a*(x**2) + b*(x) + c
    return y

def plot_V(Area, Name, X, line_fit):
    VPP = []
    AREA = []

    for i in range(len(X)):
        if X[i] == 1:
            try:
                vpp = Name[i].split("kVpp")[0].split("trek_")[-1]
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
            curr_label = curr_label[curr_label.find('\\', 16)+1:]
            plt.plot(vpp, Area[i], symb, label = curr_label)
            VPP.append(vpp)
            AREA.append(Area[i])
    if line_fit:
        popt, pcov = curve_fit(func, VPP, AREA)
        voltages = np.linspace(np.min(VPP), np.max(VPP), 100)
        plt.plot(voltages, func(voltages, *popt))
        print popt
        print pcov

    plt.xlabel("Vpp_daq [V]")
    plt.ylabel("Volts X sensor (100 gain) [V]")
    plt.legend()
    return


Area, Name = plot_all(folder_list, folder_signal)

# Area2, Name2 = plot_all(folder_list_s, folder_signal)


plt.figure()

# plot_V(Area, Name, Area_data, True)

# plot_V(Area2, Name2, Area_data2, False)

plt.grid()
plt.show()
