import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np
import bead_util as bu
import glob

read_npy = [1,1,0]  # 1 means do NOT read h5
read_signal_npy = True

Area_data = [1,1,1]  # 1 means plot

folder_signal = [r"C:\data\20181003\modulation_pickup_test_high_voltage_connected\sample"]

folder_list1 = [r"C:\data\20180806\electronic_noise_trek_4kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_0.4kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_2kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_1kVpp_AI1offdaq_laser_on",
                r"C:\data\20180806\electronic_noise_trek_3kVpp_AI1offdaq_laser_on", ]

folder_list2 = [  # r"C:\data\20180806\farther_electrodes\electronic_noise_trek_3kVpp_AI1offdaq_laser_on",
    r"C:\data\20180806\farther_electrodes\electronic_noise_trek_4kVpp_AI1offdaq_laser_on", ]
# r"C:\data\20180806\farther_electrodes\electronic_noise_trek_2kVpp_AI1offdaq_laser_on",
# r"C:\data\20180806\farther_electrodes\electronic_noise_trek_0.5kVpp_AI1offdaq_laser_on",
# r"C:\data\20180806\farther_electrodes\electronic_noise_trek_1kVpp_AI1offdaq_laser_on"]

folder_list3 = [r"C:\data\20180806\farther_electrodes\electronic_noise_trek_4kVpp_AI1offdaq_laser_off", r"C:\data\20180806\farther_electrodes\47Hz\electronic_noise_trek_4kVpp_AI1offdaq_laser_on"]

folder_list4 = [  # r"C:\data\20180806\before_chamber\vertical_electronic_noise_trek_4kVpp_AI1offdaq_laser_on",
    # r"C:\data\20180806\before_chamber\electronic_noise_trek_4kVpp_AI1offdaq_laser_off",
    # r"C:\data\20180806\before_chamber\horizontal_electronic_noise_trek_off_AI1offdaq_laser_on",
    r"C:\data\20180810\TREK_unplugged\laser_on_4kV_x", ]

folder_list5 = [r"C:\data\20180810\laser_off_4kV_old_XY",
                r"C:\data\20180810\laser_on_4kV_old_XY",
                r"C:\data\20180810\laser_on_2kV_old_XY",
                r"C:\data\20180810\laser_on_3kV_old_XY"]

folder_list6 = [r"C:\data\20180813\laser_off_4kV_1064nm_after_chamber",
                r"C:\data\20180813\laser_on_4kV_1064nm_after_chamber",
                r"C:\data\20180813\laser_on_4kV_1064nm_with_lens_after_chamber_far",
                r"C:\data\20180813\laser_on_2kV_1064nm_with_lens_after_chamber_far",
                r"C:\data\20180813\laser_on_3kV_1064nm_with_lens_after_chamber_far",
                r"C:\data\20180813\laser_on_5kV_1064nm_with_lens_after_chamber_far",
                r"C:\data\20180813\laser_on_6kV_1064nm_with_lens_after_chamber_far",]

folder_list7 = [r"C:\data\20180813\laser_on_5kV_1064nm_with_lens_after_chamber_far",
                r"C:\data\20180813\laser_on_5kV_1064nm_before_chamber",
                r"C:\data\20180813\laser_off_5kV_1064nm_before_chamber",]

folder_list8 = [r"C:\data\20180813\transfer_function_x_gain_0.01_after_chamber",
                r"C:\data\20180813\transfer_function_y_gain_0.01_after_chamber",
                r"C:\data\20180813\transfer_function_0_gain_off_after_chamber",
                r"C:\data\20180813\transfer_function_x_gain_0.01_before_chamber",
                r"C:\data\20180813\transfer_function_y_gain_0.01_before_chamber",
                r"C:\data\20180813\transfer_function_0_gain_off_before_chamber"]

folder_list9 = [r"C:\data\20180816\laser_on_5kV_1064nm_after_chamber",
                r"C:\data\20180816\laser_on_5kV_top_feedthrough_1064nm_after_chamber",
                r"C:\data\20180816\laser_on_5kV_trek_unplugged_1064nm_after_chamber"]

folder_list10 = [r"C:\data\20180816\old\laser_on_5kV_trek_unplugged_top_feedthrough_1064nm_after_chamber",
                 r"C:\data\20180816\old\laser_on_5kV_trek_unplugged_bottom_feedthrough_1064nm_after_chamber",]

folder_list11 = [r"C:\data\20180816\laser_on_5kV_trek_unplugged_bottom_feedthrough_1064nm_after_chamber",
                 r"C:\data\20180816\laser_on_5kV_trek_unplugged_top_feedthrough_1064nm_after_chamber",
                 r"C:\data\20180816\laser_on_5kV_trek_unplugged_hanging_1064nm_after_chamber",
                 r"C:\data\20180816\laser_off_5kV_trek_unplugged_hanging_1064nm_after_chamber",
                 r"C:\data\20180816\laser_on_5kV_trek_unplugged_hanging_ungrounded_1064nm_after_chamber",]

folder_list12 = [r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber",
                 #r"C:\data\20180816\laser_on_5kV_trek_on_battery_unplugged_on_roof_1064nm_after_chamber",
                 r"C:\data\20180816\laser_on_5kV_trek_off_on_roof_1064nm_after_chamber",
                 r"C:\data\20180816\laser_on_5kV_trek_on_high_voltage_off_on_roof_1064nm_after_chamber",
                 r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber_2",
                 r"C:\data\20180816\laser_on_5kV_trek_on_roof_1064nm_after_chamber_y_unplugged"]

folder_list13 = [r"C:\data\20180816\transfer_function_x_gain_0.01_after_chamber",
                 r"C:\data\20180816\transfer_function_y_gain_0.01_after_chamber",
                 r"C:\data\20180816\transfer_function_0_gain_after_chamber"]

folder_list14 = [r"C:\data\tube_test_20180906\no_laser",
                 r"C:\data\tube_test_20180906\with_no_tubes",
                 r"C:\data\tube_test_20180906\with_2tubes"]

folder_list15 = [r"C:\data\20181003\modulation_pickup_test_high_voltage_connected\no_modulation_pumped_down",
                 r"C:\data\20181003\modulation_pickup_test_high_voltage_connected\with_modulation_pumped_down"]#,
                 #r"C:\data\20181003\modulation_pickup_test_high_voltage_connected\with_modulation_pumped_down"]

folder_list16 = [r"C:\data\20181010\Lock_in\single_detector\AC_on\Modulation_on",
                 r"C:\data\20181010\Lock_in\single_detector\AC_on\Modulation_on\1"]


folder_list17 = [r"C:\data\20181015\Lock_in\10kHz_mod_10kHz_cutoff_10Vpp\Only_signal",
                 r"C:\data\20181015\Lock_in\10kHz_mod_10kHz_cutoff_10Vpp\Lock_in",    
                 r"C:\data\20181015\Lock_in\10kHz_mod_10kHz_cutoff_10Vpp\Lock_in\2",  
                 r"C:\data\20181015\Lock_in\10kHz_mod_10kHz_cutoff_10Vpp\Lock_in\3",
                 r"C:\data\20181015\Lock_in\10kHz_mod_10kHz_cutoff_10Vpp\Lock_in\4"]   
                 #Only photodiode signal, no gain
                 #Locked in. Signal gain = 5.2, ref. gain = 17.2
                 #Locked in. Signal gain = 1, ref. gain = 17.2
                 #Locked in. Signal gain = 1, ref gain = 1, amp gain = 8
                 #Locked in. Strong reference. Signal gain = 1, ref gain = 0.48, amp gain = 0.45
                 #Locked in. bandpass filter on reference

folder_list18 = [r"C:\data\20181016\Lock_in\10kHz_mod\Signal_only",
                 r"C:\data\20181016\Lock_in\10kHz_mod\External_lock_in\in_phase\0_1sec_band_filter",
                 r"C:\data\20181016\Lock_in\10kHz_mod\External_lock_in\in_phase\no_post_10_pre"]   
                

folder_list = folder_list18

folder_signal = folder_list18

Fs = 10e3  ## this is ignored with HDF5 files
NFFT = 2 ** 19


def plot_psd_folders(folder_list, channel):
    def getdata(fname, drive):
        print "Opening file: ", fname
        ## guess at file type from extension
        _, fext = os.path.splitext(fname)
        if (fext == ".h5"):
            f = h5py.File(fname, 'r')
            dset = f['beads/data/pos_data']
            dat = numpy.transpose(dset)
            # max_volt = dset.attrs['max_volt']
            # nbit = dset.attrs['nbit']
            Fs = dset.attrs['Fsamp']

            # dat = 1.0*dat*max_volt/nbit
            dat = dat * 10. / (2 ** 15 - 1)

        else:
            dat = numpy.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

        if drive:
            xpsd, freqs = matplotlib.mlab.psd(dat[:, bu.drive] - numpy.mean(dat[:, bu.drive]), Fs=Fs, NFFT=NFFT)
        else:
            xpsd, freqs = matplotlib.mlab.psd(dat[:, channel] - numpy.mean(dat[:, channel]), Fs=Fs, NFFT=NFFT)

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

        file_list = glob.glob(folder + "\*.h5")
        if len(folder) == 0:
            return "error empty folder"

        freq, p2 = getdata(file_list[0], drive)
        p2 = np.zeros(len(p2))
        for i in file_list:
            f, aux = getdata(i, drive)
            p2 = p2 + aux

        if not drive:
            if channel == bu.xi:
                np.save(os.path.join(folder, "measurement_x"), [freq, p2 / len(file_list)])
            else:
                np.save(os.path.join(folder, "measurement_y"), [freq, p2 / len(file_list)])

        return [freq, p2 / len(file_list)]

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
                np.load(os.path.join(folder[0], "position.npy"))  # the zero is because the way things are structured.
                return np.load(os.path.join(folder[0], "position.npy"))
            except:
                print "no npy"

        f, p2, name = read_psd_folders(folder, True)
        a = np.argmax(p2)
        np.save(os.path.join(folder[0], "position"), a)
        return a

    def plot_all(folder_list, folder_signal):
        arg = peak_position(folder_signal)
        F, P2, Name = read_psd_folders(folder_list, False)
        Area = []
        plt.figure()
        for i in range(len(folder_list)):
            curr_label = Name[i]
            curr_label = curr_label[curr_label.find('\\', 16) + 1:]
            plt.loglog(F[i], np.sqrt(P2[i]), label=curr_label)
            a = np.sum(P2[i][arg - 1:arg + 1])
            a = np.sqrt(a)
            Area.append(a)
        plt.axvline(F[0][arg], color="r", linestyle="--")
        plt.axvline(F[0][arg]+10000, color="g", linestyle="--")
        plt.axvline(-F[0][arg]+10000, color="g", linestyle="--")
        plt.axvline(10000, color="k", linestyle="--")
        plt.legend()
        #plt.xlim(45,51)
        plt.ylim(1e-6, 5)
        plt.xlabel("Frequency (Hz)")
        if channel == bu.xi:
            name_channel = "X"
        else:
            name_channel = "Y"
        plt.ylabel("PSD (V/$\sqrt{Hz}$)"+" of channel "+ name_channel)
        plt.grid()
        plt.tight_layout(pad = 0)

        return [Area, Name]

    def plot_V(Area, Name, X):
        plt.figure()
        for i in range(len(X)):
            if X[i] == 1:
                try:
                    vpp = Name[i].split("kV")[0]
                    vpp = vpp[vpp.rfind("_")+1:]
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
        plt.tight_layout(pad = 0)
        return
    
    Area, Name = plot_all(folder_list, folder_signal)

    # plot_V(Area, Name, Area_data)


plot_psd_folders(folder_list, bu.xi)
# plot_psd_folders(folder_list, bu.yi)
plt.show()
