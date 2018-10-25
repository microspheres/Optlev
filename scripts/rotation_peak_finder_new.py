import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import bead_util as bu
import glob
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
import cPickle as pickle


def list_file_time_order(filelist):
    filelist.sort(key=os.path.getmtime)
    return filelist

press_fit_linear = False
press_fit_exp = False
press_smooth = False
fit = False
damping_plot = False
remake_file = True
file_jump = 1
power_spectrum_fluctuations_res = False
ylim = [0.01, 4000]
good_time = 2500.
k = 3 # smooth order
NFFT = 2**17
ss = 3. ## std for the goodpoints
first_peak = True
torque = False

rho = 1800. # kg/m^3
radius = 15.0*1e-6 #m #22um sphere



parameters = [1370000., 0.1, 5000, "down", NFFT]  # last entry shoulp help with aliasing. Gets "up", "down" or "none". "up" for curves that go up," down" for curves that go down and "none" for nothing to happens.


path_list = [r"C:\data\20180323\bead3_SiO2_15um_POL_NS\meas6_spin_scan"]



file_list1 = []
for path1 in path_list:

    file_list = glob.glob(path1+"\*.h5")
    file_list = list_file_time_order(file_list)
    file_list1 += file_list[::file_jump]


def getdata(fname, NFFT):
	print "Opening file: ", fname
	## guess at file type from extension
	_, fext = os.path.splitext( fname )
	if( fext == ".h5"):
		f = h5py.File(fname,'r')
		dset = f['beads/data/pos_data']
		dat = numpy.transpose(dset)
		#max_volt = dset.attrs['max_volt']
		#nbit = dset.attrs['nbit']
		Fs = dset.attrs['FS_scope']
                Press = dset.attrs['temps'][0]
                time = dset.attrs['Time']
		
		#dat = 1.0*dat*max_volt/nbit
                dat = dat * 10./(2**15 - 1)
                
	else:
		dat = numpy.loadtxt(fname, skiprows = 5, usecols = [2] )

        xpsd_old, freqs = matplotlib.mlab.psd(dat[:, -1]-numpy.mean(dat[:, -1]), Fs = Fs, NFFT = NFFT)
        # Ddrive = dat[:, bu.drive]*np.gradient(dat[:,bu.drive])
        # DdrivePSD, freqs =  matplotlib.mlab.psd(Ddrive-numpy.mean(Ddrive), Fs = Fs, NFFT = NFFT)

        #for h in [xpsd, ypsd, zpsd]:
        #        h /= numpy.median(dat[:,bu.zi])**2
	return [freqs, 0, 0, 0, 0, xpsd_old, Press, time]


def return_arg(list, value):
    aux = 1.0*np.array(list) - 1.0*value
    aux = np.abs(aux)
    arg = np.argmin(aux)
    return arg

N = 1
def finder(filelist, para):
    directory = os.path.split(filelist[0])[0]
    savefile = os.path.join(directory, "savedata.txt")
    if os.path.isfile(savefile) and not remake_file:
        sf = open(savefile, "rb")
        data = pickle.load(sf)
        sf.close()
        return data


    peak_pos, peaks_distance, peak_step, aliasing, NFFT = para
    freq = getdata(filelist[0], NFFT)[0]

    if aliasing == "up":
        up_aux = 0.1
        down_aux = 1.0
    else:
        if aliasing == "down":
            down_aux = 0.1
            up_aux = 1.0
        else:
            if aliasing == "none":
                up_aux = 1.0
                down_aux = 1.0

    rotation = []
    P = []
    T = []

    last_peak_pos = peak_pos
    last_step = peak_step

    for i in range(len(filelist)/N):
        
        data = np.zeros(len(freq))

        for j in range(N):
            gd = getdata(filelist[i*N + j], NFFT)
            data += gd[5]
        data /= N

        W = peaks_distance*peak_pos
        argpeak = return_arg(freq, peak_pos)
        argW = return_arg(freq, W)
        
        Peakrange = data[(argpeak - int(up_aux*argW)):(argpeak + int(down_aux*argW))]

        shortfreq = freq[(argpeak - int(up_aux*argW)):(argpeak + int(down_aux*argW))]
        
        lin = np.polyfit(shortfreq, Peakrange, 1)

        subtract_data = Peakrange - np.polyval(lin, shortfreq)
        Peak = np.argmax(subtract_data)
        
        if False:
            plt.figure()
            plt.plot(shortfreq,subtract_data)
            plt.plot(shortfreq, 5*np.ones(len(shortfreq))*np.std(subtract_data))
            plt.show()
        
        #if (len( np.argwhere(subtract_data > 7.*np.std(subtract_data))) > 1):
        #if ( i>0 and  np.abs(shortfreq[Peak] - peak_pos) > 50*np.abs(peak_pos - last_peak_pos) ):
        #    rot = -1
        #else:
        rot = shortfreq[Peak]
        

        if False and i > 140:
            plt.figure()
            plt.loglog(freq,data)
            plt.plot(shortfreq, data[(argpeak - argW):(argpeak + argW)])
            plt.plot(shortfreq[Peak], data[(argpeak - argW):(argpeak + argW)][Peak], "rx")
            #plt.xlim([peak_pos - 2*peaks_distance, peak_pos + 2*peaks_distance])
            plt.show()

        if(i > 0 ):
            curr_step = rot-last_peak_pos
        else:
            curr_step = 10*last_step
        
        if np.abs(curr_step) < 5*np.abs(last_step):
            peak_pos = rot
            last_step = curr_step
            last_peak_pos = rot
        else:
            peak_pos = rot #+= last_step
            last_peak_pos = peak_pos

        # print peak_pos, rot, last_peak_pos

        #print peak_pos, last_peak_pos, rot     
        #if rot > 0:
        #    if i>0:
        #        last_peak_pos = rot
        #else:
        #    temp_pos = last_peak_pos
        #    peak_pos = 2.*peak_pos - temp_pos
        #    last_peak_pos = peak_pos
        #print peak_pos, last_peak_pos, rot
        #raw_input()
            
        rotation.append(rot)
        P.append(gd[6])
        T.append(gd[7])
    # return [np.array(rotation), np.array(P), np.array(T) - T[0]]
    data = [rotation, P, np.array(T)]
    sf = open(savefile, "wb")
    pickle.dump(data, sf)
    sf.close()
    # return [np.array(rotation), np.array(P), np.array(T) - T[0]]
    return [rotation, P, np.array(T)]

c1 = finder(file_list1, parameters)



t0 = c1[2][0]
t = np.hstack([c1[2]])
rotation = c1[0]
pressures = c1[1]


if first_peak:
    div = 1.
else:
    div = 2.

c1 = [np.array(rotation)/div, np.array(pressures), t - t0]






# bad points off

aux = np.diff(c1[0])

aux = np.hstack([aux, 0])

aux = np.roll(aux, 1)

mean, std = np.mean(aux), np.std(aux)

good_points = np.abs(aux-mean) < ss*std

good_points1 = np.logical_and(good_points, c1[2] > good_time)



################### fitting the pressure with a line

def pressline(x, a, b):
    y = a*x + b
    return y

def exp(x, x0, a, b, c):
    y = a*np.exp(c*(x-x0)) + b
    return y




#  fitting


def func(x, x0, A, tau):
    f = A*(1.0 - np.exp(-(x-x0)/tau))
    return f

def func2(X, x0, A, B): # takes pressure into account
    x, p = X
    f = (A/p)*(1.0 - np.exp(-(x-x0)*B*p))
    return f

def funcdecay(x, x0, A, tau, v0):
    f = A + v0*np.exp(-(x-x0)/tau)
    return f

def funcdecay2(X, x0, A, B, v0): # takes pressure into account
    x, p = X
    f = (A/p) + v0*np.exp(-(x-x0)*B*p)
    return f




### making the line fit for the pressure

if press_fit_linear:
    print "mean pressure:", np.mean(c1[1][good_points1])
    poptpr, pcovpr = curve_fit(pressline, c1[2][good_points1], c1[1][good_points1])
    press = pressline(c1[2][good_points1], *poptpr)
else:
    ### pressure smooth
    if press_smooth:
        smooth = UnivariateSpline(c1[2][good_points1], c1[1][good_points1], k = k)
        # smooth.set_smoothing_factor(4)
        press = smooth(c1[2][good_points1])
    else:
        if press_fit_exp:
            poptpr, pcovpr = curve_fit(pressline, c1[2][good_points1], np.log(c1[1][good_points1]))
            press = np.exp(pressline(c1[2][good_points1], *poptpr))
        else:
            press = c1[1][good_points1]






###


p0 = [ -1.6e+03,   1.2e+11,   5.6e+08]

p02 = [ -1.3e+03,   1.08e+00,   2.379e+02]



if fit:
    popt, pcov = curve_fit(func, c1[2][good_points1], c1[0][good_points1], p0 = np.array(p0), sigma = 5.E6/(NFFT/2))
    #popt2, pcov2 = p02, np.zeros([len(p02),len(p02)])
    popt2, pcov2 = curve_fit(func2, (c1[2][good_points1], press), c1[0][good_points1], p0 = np.array(p02), sigma = 5.E6/(NFFT/2))
    print popt2

else:
    popt = p0
    pcov = np.zeros([len(p0),len(p0)])
    popt2 = p02
    pcov2 = np.zeros([len(p02),len(p02)])

    



print "pressure not included up"
print "read: time0, final speed, tau"
print popt
print pcov
# tauP = 1/(popt2[2]*np.mean(c1[1][good_points1]))
# print "####################################"
# print "pressure included down, <tau> is:", tauP
# print "dtau:", tauP*np.sqrt( pcov2[2][2]/(popt2[2]**2) +  (np.std(c1[1][good_points1])**2)/(np.mean(c1[1][good_points1]))**2)
# print "final speed:", popt2[1]/np.mean(c1[1][good_points1])
# print "####################################"
# print "pressure not included up"
# print popt3
# print pcov3
# print "####################################"
tauP2 = 1/(popt2[2]*np.mean(c1[1][good_points1]))
print "pressure included up, <tau> is", tauP2
print "dtau:", tauP2*np.sqrt( pcov2[2][2]/(popt2[2]**2) +  (np.std(c1[1][good_points1])**2)/(np.mean(c1[1][good_points1]))**2)



if torque:
    Iner = (8./15.)*rho*np.pi*(radius**5)
    print "Torque[N.m]", Iner*popt[1]/popt[2]



print "mean pressure of good points:", np.mean(c1[1][good_points1])

times =  np.linspace(-100, 20000, 20000)


# cmap=plt.get_cmap('cool')
plt.figure()

plt.scatter(c1[2], c1[0]/1000., s=8, c=np.log10(c1[1]))

plt.plot(times, func(times, *popt)/1000., "r--", lw = 1, label = "Fit with no pressure")

plt.plot(c1[2][good_points1], func2((c1[2][good_points1], press), *popt2)/1000., "k--", lw = 1, label = "Fit with pressure")


plt.grid()
plt.ylim(ylim[0],ylim[1])

plt.ylabel("Rotation [kHz]", fontsize=13)
plt.xlabel("Time [s]", fontsize=13)
# plt.plot(c1[2][good_points1], c1[0][good_points1], color = "g")
plt.colorbar()
plt.legend(loc="lower right", frameon = False)
plt.tight_layout(pad = 0)

plt.figure()

#resi

plt.scatter(c1[2][good_points1], (func(c1[2][good_points1], *popt) -  c1[0][good_points1])/c1[0][good_points1], s=8, c=np.log10(c1[1][good_points1]), label = "Fit with no pressure")

plt.plot(c1[2][good_points1], (func2((c1[2][good_points1], press), *popt2) -  c1[0][good_points1])/c1[0][good_points1], "kx", label = "Fit with pressure")


plt.ylabel("dw/w", fontsize=13)
plt.xlabel("Time [s]", fontsize=13)
plt.colorbar()
plt.grid()
plt.legend(loc="upper right", frameon = False)
plt.tight_layout(pad = 0)

#pressure
plt.figure()
plt.plot(c1[2], c1[1], "r-x",label = "pressure")
plt.plot(c1[2][good_points1], press, "b-",label = "pressure fit")

plt.ylabel("Pressure [mbar]", fontsize=13)
plt.xlabel("Time [s]", fontsize=13)

plt.grid()
plt.legend(loc="upper right", frameon = False)
plt.tight_layout(pad = 0)

# damping vs pressure
if damping_plot:
    
    damping_time =  1./(popt2[2]*c1[1][good_points1])

    plt.figure()
    plt.loglog(c1[1][good_points1], damping_time, "r.")


if power_spectrum_fluctuations_res:
    psd = np.abs(np.fft.rfft(c1[1] - np.mean(c1[1])))
    freq = np.fft.rfftfreq(len(c1[1]), 1./(c1[2][1] - c1[2][0]))
    plt.figure()
    plt.loglog(freq, psd)


plt.show()



# plt.yticks(fontsize=12)
# plt.xticks(fontsize=12)
# plt.tight_layout(pad = 0)






# path_list = [r"C:\data\20180129\bead1_um_POL_NS_SiO2_10um\meas_1_V-3", 
#              r"C:\data\20180129\bead1_um_POL_NS_SiO2_10um\meas_2_V+1", 
#              r"C:\data\20180129\bead1_um_POL_NS_SiO2_10um\meas_3_V-2"]

# p0 = [ 2768., 260513.,   15177.,  469189.]
# p03 = [ 27600., 1E6,   15177.]
# p02 = [ 2800, 260513.*(3E-7),   1/(15177.*(3E-7)),  469189.]

# if False:
#     popt, pcov = curve_fit(funcdecay, c1[2][good_points1], c1[0][good_points1], p0 = np.array(p0), sigma = 1.E6/(NFFT/2))



#     popt2, pcov2 = curve_fit(funcdecay2, (c1[2][good_points1], c1[1][good_points1]), c1[0][good_points1], p0 = np.array(p02), sigma = 1.E6/(NFFT/2))


#     popt3, pcov3 = curve_fit(func, c1[2][good_points2], c1[0][good_points2], p0 = np.array(p03), sigma = 1.E6/(NFFT/2))


# c1 = finder(file_list1, 9.99E+5, 8.E4, 1.E4)
# c2 = finder(file_list2, 4.17786E+6, 3.E4, 1.E4)
# c3 = finder(file_list3, 5.47769E+6, 3.E4, 1.E4)


# t0 = c1[2][0]
# t = np.hstack([c1[2],c2[2],c3[2]])
# rotation = c1[0] + c2[0] + c3[0]
# pressures = c1[1] + c2[1] + c3[1]
