import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
import numpy as np

import glob
import scipy.optimize as opt

rho = 1800.0

R = (5.0e-6)

M = (4./3.)*np.pi*(R**3*rho)
dR=(0.7/2)*10**(-6)
dM=M*np.sqrt((dR/R)**2)

electron = 1.60218e-19

kb = 1.38e-23

f_crit=1800

acceleration_plot = False

HP_plot=False # Whether or not to show HP data on plots

no_sphere = True
pathno = [r"C:\data\20191022\10um\prechamber_LP\1\nosphere",]

distance = 0.0021
distance_error = 1e-4

NFFT = 2**16

#Make sure that this is actually the right calibration path
path_calibration = r"C:\data\20191022\10um\prechamber_LP\1\calibration1p"

path_list_temp = [r"C:\data\20191022\10um\prechamber_LP\1\temp_x\1", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\2", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\3", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\4", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\5", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\6", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\7", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\8", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\9", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\10", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\11", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\13", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\14", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\15", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\16", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\17", r"C:\data\20191022\10um\prechamber_LP\1\temp_x\18"]


path_high_pressure_nofb = r"C:\data\20191022\10um\prechamber_LP\1\2mbar"
file_high_pressure_nofb = "2mbar_zcool.h5"

f_start = 50.  # for the fit
f_end = 110.  # for the fit

delta = 1e-2
fq = np.arange(f_start, f_end, delta)


def getdata(fname):
    print
    "Opening file: ", fname
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
        PID = dset.attrs['PID']
        press = dset.attrs['pressures']

    else:
        dat = numpy.loadtxt(fname, skiprows=5, usecols=[2, 3, 4, 5, 6])

    xpsd, freqs = matplotlib.mlab.psd(dat[:, 0] - numpy.mean(dat[:, 0]), Fs=Fs, NFFT=NFFT)
    fieldpsd, freqs = matplotlib.mlab.psd((dat[:, 3] - numpy.mean(dat[:, 3])), Fs=Fs, NFFT=NFFT)

    return [freqs, xpsd, PID, press[0], fieldpsd]


def get_high_pressure_psd(path_hp, file_hp):
    a = getdata(os.path.join(path_hp, file_hp))
    freq = a[0]
    xpsd = a[1]
    return [freq, xpsd]


def get_files_path(path):
    file_list = glob.glob(path + "\*.h5")
    return file_list


def get_data_path(path):  # PSD output is unit square, V**2/Hz : it assumes that within the folder, Dgx is the same.
    info = getdata(get_files_path(path)[0])
    freq = info[0]
    dfreq=freq[1]-freq[0]
    dgx = info[2][0]
    Xpsd = np.zeros(len(freq))
    fieldpsd = np.zeros(len(freq))
    aux = get_files_path(path)
    for i in aux:
        a = getdata(i)
        Xpsd += a[1]
        fieldpsd += a[4]
        p = a[3]
    Xpsd = Xpsd / len(aux)
    fieldpsd = fieldpsd / len(aux)
    return [Xpsd, dgx, p, fieldpsd, freq,dfreq]


def plot_psd(path):
    a = get_data_path(path)
    freq = a[4]
    plt.figure()
    plt.loglog(freq, a[0])
    plt.loglog(freq, a[3])
    return "hi!"


def findAC_peak(path):
    a = get_data_path(path)
    freq = a[4]
    pos = np.argmax(a[3])
    return [pos, freq[pos]]


def get_field(path):
    a = get_data_path(path)
    pos = findAC_peak(path)[0]
    v = np.sum(a[3][pos - 3:pos + 3])*a[5]
    v_amp =200. *  np.sqrt(v)*np.sqrt(2.)
    E_amp = v_amp / distance
    return [v_amp, E_amp]


def force1e(path):  # gives force of 1e of charge
    E = get_field(path)[1]
    F = E * electron
    return F


def acc(path):  # gives the acc of 1e of charge
    F = force1e(path)
    acc = F / M
    return acc


def get_sensor_motion_1e(path):
    pos = findAC_peak(path)[0]
    a = get_data_path(path)
    sen = np.sum(a[0][pos - 3:pos + 3])*a[5]
    sen_amp = np.sqrt(sen) *np.sqrt(2.)
    return sen_amp


def psd(f, A, f0, gamma):
    w0 = 2. * np.pi * f0
    w = 2. * np.pi * f
    gamma1 = 2.0 * np.pi * gamma
    s1 = 2. * (gamma1 * (w0 ** 2))
    s2 = 1. * (w0 ** 2) * ((w0 ** 2 - w ** 2) ** 2 + (gamma1 * w) ** 2)
    s = np.sqrt(s1 / s2)
    return A * s


def fit_high_pressure_no_fb(path_hp, file_hp):
    a = get_high_pressure_psd(path_hp, file_hp)
    freq = a[0]
    xpsd = np.sqrt(a[1])
    fit_points1 = np.logical_and(freq > f_start, freq < 59.0)
    fit_points2 = np.logical_and(freq > 61.0, freq < 119.0)
    fit_points3 = np.logical_and(freq > 121.0, freq < 122.0)
    fit_points4 = np.logical_and(freq > 123.3, freq < 144.8)
    fit_points5 = np.logical_and(freq > 145.9, freq < 179.0)
    fit_points6 = np.logical_and(freq > 181.0, freq < f_end)
    fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6
    p0 = [0.1, 70, 100.]
    popt, pcov = opt.curve_fit(psd, freq[fit_points_new], xpsd[fit_points_new], p0=p0)
    freqplot = fq
    # plt.figure()
    # plt.loglog(freq, xpsd)
    # plt.loglog(freqplot, psd(freqplot, *popt))
    return [popt, freq, freqplot, xpsd]


def convert_sensor_meter(path, path_hp,
                         file_hp):  # given that the feedback barelly affects the motion due to the ac field
    sen_amp = get_sensor_motion_1e(path)
    acc1e = acc(path)
    f0 = fit_high_pressure_no_fb(path_hp, file_hp)[0][1]
    motiontheo = 1.0 * acc1e / ((2.0 * np.pi * f0) ** 2)
    C = 1.0 * motiontheo / sen_amp
    return C


def tempeture_path(path, path_hp, file_hp, pathcharge):
    a = get_data_path(path)
    xpsd = np.sqrt(a[0])
    dgx = a[1]
    freq = a[4]
    Conv = convert_sensor_meter(pathcharge, path_hp, file_hp)
    b = fit_high_pressure_no_fb(path_hp, file_hp)[0]
    f0 = b[1]

    fit_points1 = np.logical_and(freq > f_start, freq < 39.9)
    fit_points2 = np.logical_and(freq > 41.2, freq < 42.6)
    fit_points3 = np.logical_and(freq > 43.9, freq < 56.5)
    fit_points4 = np.logical_and(freq > 57.6, freq < 58.2)
    fit_points5 = np.logical_and(freq > 60.6, freq < 69.4)
    fit_points6 = np.logical_and(freq > 70.3, freq < 71.9)
    fit_points7 = np.logical_and(freq > 73.1, freq < 78.9)
    fit_points8 = np.logical_and(freq > 81.6, freq < 83.5)
    fit_points9 = np.logical_and(freq > 88.8, freq < 89.7)
    fit_points10 = np.logical_and(freq > 90.4, freq < 92.9)
    fit_points11 = np.logical_and(freq > 93.8, freq < f_end)
    
    fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11
    p0 = [1e-1, np.abs(f0), 100.]
    popt, pcov = opt.curve_fit(psd, freq[fit_points_new], xpsd[fit_points_new], p0=p0)

    f = fq
    aux = (2. * np.pi * np.abs(f0)) * Conv * psd(f, *popt)
    tempaux = np.sum(aux ** 2) * delta
    tempaux = 0.5 * M * tempaux
    temp = 2*tempaux / kb #factor 2 is to account for spring energy
    return [temp, dgx, popt, freq, xpsd]


def normsquaredH(f,a):
    w=2*np.pi*f
    wc=2*np.pi*f_crit
    tranhigh=w/(w+wc)
    tranlow=1/(1+w/wc)
    totaltran=a*tranhigh*tranlow
    return totaltran**2

def Hphaseshift(f):
    w=2*np.pi*f
    wc=2*np.pi*f_crit
    shiftlow=-np.arctan(w/wc)
    shifthigh=np.pi/2-np.arctan(w/wc)
    totalshift=shiftlow+shifthigh
    return totalshift


def RealH(f,a):
    return np.sqrt(normsquaredH(f,a))*np.cos(Hphaseshift(f))

def ImH(f,a):
    return np.sqrt(normsquaredH(f,a))*np.sin(Hphaseshift(f))


def noisepsd(fdgxnoise, A, f0, gamma, amp, a):
    f=fdgxnoise[0]
    dfb=fdgxnoise[1]
    noise=fdgxnoise[2]
    w0 = 2. * np.pi * f0
    w = 2. * np.pi * f
    gamma1 = 2.0 * np.pi * gamma

    numterm1=gamma1*amp ** 2
    numterm2=1 #2*amp*np.sqrt(gamma1)*(w0**2-w**2)*noise
    numterm3=((w0 ** 2 - w ** 2) ** 2+(gamma1 ** 2) * (w ** 2)) * (noise ** 2)
    num=numterm1+numterm2+numterm3

    denterm1=(w0 ** 2 - w ** 2) ** 2
    denterm2=2. * RealH(f,a) * dfb * (w0 ** 2) * (w0 ** 2 - w ** 2)
    denterm3=2*gamma1*w*(w0**2)*dfb*ImH(f,a)
    denterm4=(dfb ** 2) * (w0 ** 4) * normsquaredH(f,a)
    denterm5=(gamma1 ** 2) * (w ** 2)
    den=denterm1-denterm2-denterm3+denterm4+denterm5

    s=np.sqrt(num/den)

    return A * s


def getnoiselevel(pathno,path_hp,file_hp,pathcharge):
    a=tempeture_path(pathno,path_hp,file_hp,pathcharge)
    xpsd=a[4]**2
    freq=a[3]

    fit_points1 = np.logical_and(freq > f_start, freq < 39.9)
    fit_points2 = np.logical_and(freq > 41.2, freq < 42.6)
    fit_points3 = np.logical_and(freq > 43.9, freq < 56.5)
    fit_points4 = np.logical_and(freq > 57.6, freq < 58.2)
    fit_points5 = np.logical_and(freq > 60.6, freq < 69.4)
    fit_points6 = np.logical_and(freq > 70.3, freq < 71.9)
    fit_points7 = np.logical_and(freq > 73.1, freq < 78.9)
    fit_points8 = np.logical_and(freq > 81.6, freq < 83.5)
    fit_points9 = np.logical_and(freq > 88.8, freq < 89.7)
    fit_points10 = np.logical_and(freq > 90.4, freq < 92.9)
    fit_points11 = np.logical_and(freq > 93.8, freq < f_end)
    
    fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11

    noise=np.sqrt(np.mean(xpsd[fit_points_new]))

    return noise


def noisepsd_fit_path(path,pathno,path_hp,file_hp,pathcharge):
    a = get_data_path(path)
    xpsd = np.sqrt(a[0])
    dgx = a[1]
    freq = a[4]
    Conv = convert_sensor_meter(pathcharge, path_hp, file_hp)
    singlenopath=pathno[0]
    noise=getnoiselevel(singlenopath,path_hp,file_hp,pathcharge)
    b = fit_high_pressure_no_fb(path_hp, file_hp)[0]
    f0 = b[1]
    gammaguess=b[2]

    dgxarray=dgx*np.ones(len(freq))
    noisearray=noise*np.ones(len(freq))
    fdgxnoise=np.transpose((freq,dgxarray,noisearray))

    fit_points1 = np.logical_and(freq > f_start, freq < 39.9)
    fit_points2 = np.logical_and(freq > 41.2, freq < 42.6)
    fit_points3 = np.logical_and(freq > 43.9, freq < 56.5)
    fit_points4 = np.logical_and(freq > 57.6, freq < 58.2)
    fit_points5 = np.logical_and(freq > 60.6, freq < 69.4)
    fit_points6 = np.logical_and(freq > 70.3, freq < 71.9)
    fit_points7 = np.logical_and(freq > 73.1, freq < 78.9)
    fit_points8 = np.logical_and(freq > 81.6, freq < 83.5)
    fit_points9 = np.logical_and(freq > 88.8, freq < 89.7)
    fit_points10 = np.logical_and(freq > 90.4, freq < 92.9)
    fit_points11 = np.logical_and(freq > 93.8, freq < f_end)
    
    fit_points_new = fit_points1 + fit_points2 + fit_points3 + fit_points4 + fit_points5 + fit_points6 + fit_points7 + fit_points8 + fit_points9 + fit_points10 + fit_points11

    selectfreq=freq[fit_points_new]
    selectdgx=dgx*np.ones(len(selectfreq))
    selectnoise=noise*np.ones(len(selectfreq))
    selectpsd=xpsd[fit_points_new]
    fitfdgxnoise=(selectfreq, selectdgx, selectnoise)

    p0 = [1e-3, np.abs(f0), gammaguess, 1e5, 1]  # 0th positon is A, 1st is f0, 2nd is gamma, 3rd is amp, 4th is a
    popt, pcov = opt.curve_fit(noisepsd, fitfdgxnoise, selectpsd, p0=p0,bounds=([1e-15,70,0,0,0],[1e1,150,1e3,1e10,1e5]),maxfev=5000)

    f = fq
    tempdgx=dgx*np.ones(len(f))
    tempnoise=dgx*np.ones(len(f))
    tempfdgxnoise=(f,tempdgx,tempnoise)
    aux = (2. * np.pi * np.abs(f0)) * Conv * noisepsd(tempfdgxnoise, *popt)
    tempaux = np.sum(aux ** 2) * delta
    tempaux = 0.5 * M * tempaux
    temp = 2*tempaux / kb #factor 2 is to account for spring energy

    return [temp,dgx,popt,freq,fdgxnoise,xpsd]


def noise_temp_bound(f,f0,dfb,gamma,a):
    w=2*np.pi*f
    w0=2*np.pi*f0
    gamma1=2*np.pi*gamma

    s1=(w0**2)*dfb
    Hsquarednorm=normsquaredH(f,a)
    s2=(w0**2-w**2)**2+(gamma1**2)*(w**2)
    initbound=s1*np.sqrt(Hsquarednorm/s2)
    bound=max(1,initbound)
    return [bound, initbound]


def full_noise_temp_bound(f,f0,dfb,gamma,a,amp,noise):
    w = 2 * np.pi * f
    w0 = 2 * np.pi * f0
    gamma1 = 2 * np.pi * gamma
    Hsquarednorm = normsquaredH(f, a)
    ReH=RealH(f,a)
    lamd=amp*np.sqrt(gamma1)

    numterm1=lamd**2
    numterm2=2*lamd*(w0**2)*dfb*ReH*noise
    numterm3=(w0**4)*(dfb**2)*Hsquarednorm*(noise**2)
    num=numterm1+numterm2+numterm3

    denterm1=lamd**2
    denterm2=2*lamd*(w0**2-w**2)*noise
    denterm3=((w0**2-w**2)**2+(w**2)*(gamma1**2))*(noise**2)
    den=denterm1+denterm2+denterm3

    initbound=np.sqrt(num/den)
    bound=max(1,initbound)

    return [bound,initbound]


def noise_temp_bound_path(path,pathno,path_hp,file_hp,pathcharge):
    noisefit=noisepsd_fit_path(path,pathno,path_hp,file_hp,pathcharge)
    noisefitparams=noisefit[2]
    noisegamma=noisefitparams[2]
    amp=noisefitparams[3]
    a=noisefitparams[4]
    dfb=noisefit[1]

    fdgxnoise=noisefit[4]
    noise=fdgxnoise[0][2]

    Conv=convert_sensor_meter(pathcharge, path_hp, file_hp)

    normfit=tempeture_path(path,path_hp,file_hp,pathcharge)
    b = fit_high_pressure_no_fb(path_hp, file_hp)[0]
    f0 = b[1]

    f = fq
    aux = (2. * np.pi * np.abs(f0)) * Conv * psd(f, *normfit[2])
    aux2=np.zeros(len(aux))
    aux3 = np.zeros(len(aux))
    freqbounds=np.zeros(len(aux))
    fullfreqbounds = np.zeros(len(aux))
    for i in range(0,len(aux2)):
        upbound=noise_temp_bound(f[i],f0,dfb,noisegamma,a)
        aux2[i]=aux[i]*upbound[0]
        freqbounds[i]=upbound[1]

        fullupbound = full_noise_temp_bound(f[i], f0, dfb, noisegamma, a,amp,noise)
        aux3[i] = aux[i] * fullupbound[0]
        fullfreqbounds[i] = fullupbound[1]
    tempaux=np.sum(aux**2)*delta
    tempaux=0.5 * M *tempaux

    tempboundaux = np.sum(aux2 ** 2) * delta
    tempboundaux = 0.5 * M * tempboundaux

    tempfullboundaux = np.sum(aux3 ** 2) * delta
    tempfullboundaux = 0.5 * M * tempfullboundaux

    temp=2*tempaux/kb
    tempbound = 2*tempboundaux / kb #factor 2 is to account for spring energy
    tempfullbound=2*tempfullboundaux/kb

    maxbound=max(freqbounds)
    maxfullbound=max(fullfreqbounds)

    return [temp, tempbound,maxbound,tempfullbound,maxfullbound,noise]


def noise_squash_level_path(path,pathno,path_hp,file_hp,pathcharge):
    fit=noisepsd_fit_path(path,pathno,path_hp,file_hp,pathcharge)
    f0=fit[2][1]
    w0=2*np.pi*f0
    gamma=fit[2][2]
    gamma1=2*np.pi*gamma
    amp=fit[2][3]
    a=fit[2][4]

    singlenopath = pathno[0]
    noise=getnoiselevel(singlenopath,path_hp,file_hp,pathcharge)

    num=gamma1*(gamma1*noise*np.sqrt(gamma1*(amp**2)+(gamma1**2)*(w0**2)*(noise**2))+gamma1*(amp**2)+(gamma1**2)*(w0**2)*(noise**2))
    den=gamma1*(amp**2)*w0

    squash_level=num/den


    return squash_level


def temp_path_list(pathlist, path_hp, file_hp, pathcharge, pathno, acc):
    T = []
    dT=[]
    Tbound=[]
    Tfullbound=[]
    testT=[]
    maxbounds=[]
    fullmaxbounds=[]
    noiselevels=[]
    Dgx = []
    squash_levels=[]
    f = fq
    hp = fit_high_pressure_no_fb(path_hp, file_hp)
    Conv = convert_sensor_meter(pathcharge, path_hp, file_hp)
    plt.figure()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\sqrt{S} [ m/\sqrt{Hz} ]$")
    plt.loglog(hp[1], Conv * hp[3])
    labelhp = " $\Gamma/2\Pi$ = " + str("%.1E" % hp[0][2]) + " Hz"
    plt.loglog(hp[2], Conv * psd(hp[2], *hp[0]), "k", label=labelhp)

    if no_sphere:
        ns = tempeture_path(pathno[0], path_hp, file_hp, pathcharge)
        plt.loglog(ns[3], Conv * ns[4], label="No Sphere")

    for i in pathlist:
        a = tempeture_path(i, path_hp, file_hp, pathcharge)
        dgx = a[1]
        t = a[0]
        dt=t*np.sqrt(((dM/M)**2+(2.*distance_error/distance)**2))
        b = noise_temp_bound_path(i, pathno, path_hp, file_hp, pathcharge)
        tbound = b[1]
        tfullbound=b[3]
        testt = b[0]
        maxbound = b[2]
        fullmaxbound=b[4]
        noise=b[5]
        squash_level=noise_squash_level_path(i, pathno, path_hp, file_hp, pathcharge)
        if i[:-2]=="HP":
            if HP_plot:
                T.append(t)
                dT.append(dt)
                Tbound.append(tbound)
                Tfullbound.append(tfullbound)
                testT.append(testt)
                maxbounds.append(maxbound)
                fullmaxbounds.append(fullmaxbound)
                noiselevels.append(noise)
                Dgx.append(dgx)
                squash_levels.append(squash_level)
                print
                "resonace freq =", a[2][1]
                label = " $\Gamma/2\Pi$ = " + str("%.1E" % a[2][2]) + " Hz"
                plt.loglog(a[3], Conv * a[4])
                plt.loglog(f, Conv * psd(f, *a[2]), label=label)
                plt.xlim(10, 200)
                plt.ylim(1e-13, 1e-7)
        else:
            T.append(t)
            dT.append(dt)
            Tbound.append(tbound)
            Tfullbound.append(tfullbound)
            testT.append(testt)
            maxbounds.append(maxbound)
            fullmaxbounds.append(fullmaxbound)
            noiselevels.append(noise)
            Dgx.append(dgx)
            squash_levels.append(squash_level)
            print
            "resonace freq =", a[2][1]
            label = " $\Gamma/2\Pi$ = " + str("%.1E" % a[2][2]) + " Hz"
            plt.loglog(a[3], Conv * a[4])
            plt.loglog(f, Conv * psd(f, *a[2]), label=label)
            plt.xlim(10, 200)
            plt.ylim(1e-13, 1e-7)
    plt.legend(loc=3)
    plt.grid()
    plt.tight_layout(pad=0)

    if acc:  # only to know the acc sensitivity
        C = (2.0 * np.pi * hp[0][1]) ** 2
        plt.figure()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("m/s**2/sqrt(Hz)")
        plt.loglog(hp[1], C * Conv * hp[3])
        plt.loglog(hp[2], C * Conv * psd(hp[2], *hp[0]))
        if no_sphere:
            ns = tempeture_path(pathno[0], path_hp, file_hp, pathcharge)
            plt.loglog(ns[3], C * Conv * ns[4], label="No Sphere")
        for i in pathlist:
            a = tempeture_path(i, path_hp, file_hp, pathcharge)

            plt.loglog(a[3], C * Conv * a[4])
            plt.loglog(f, C * Conv * psd(f, *a[2]))
        plt.xlim(1, 500)
        # plt.ylim(1e-13, 1e-7)
        plt.legend(loc=3)
        plt.grid()
        plt.tight_layout(pad=0)




    print Dgx
    plt.figure()
    plt.errorbar(np.array(np.abs(Dgx)) + 1e-6,1e6*np.array(T),yerr=1e6*np.array(dT),fmt="ro",label="Detected Temperature")
    plt.errorbar(np.array(np.abs(Dgx)) + 1e-6, 1e6 * np.array(Tbound), yerr=1e6*np.array(dT), color="blue", marker=".", linestyle="None",
                 label="Model Upper Bound")
    #plt.errorbar(Dgx, 1e6 * np.array(Tfullbound), yerr=np.zeros(len(Tbound)), color="green", marker=".", linestyle="None",label="Upper Bound on Temperature (8)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Feedback Gain [Arbitrary Units]",fontsize=18)
    plt.ylabel("Temperature [uK]",fontsize=18)
    plt.legend(loc=3,prop={"size":16})
    plt.grid()
    plt.tight_layout(pad=0)
    # plt.ylim(8*1e1,2*1e6)


    #Plot with noise PSD fits
    plt.figure()
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\sqrt{S} [ m/\sqrt{Hz} ]$")
    for i in pathlist:
        a=noisepsd_fit_path(i,pathno,path_hp,file_hp,pathcharge)
        dfb=a[4][0][1]
        noise=a[4][0][2]
        fitdgx=dfb*np.ones(len(f))
        fitnoise=noise*np.ones(len(f))
        plotfdgxnoise=(f,fitdgx,fitnoise)
        plt.loglog(a[3], Conv * a[5])
        plt.loglog(f, Conv * noisepsd(plotfdgxnoise, *a[2]))
        plt.xlim(10, 200)
        plt.ylim(1e-13, 1e-7)
        #print(a[2])
        plt.grid()
        plt.tight_layout(pad=0)

    #print(maxbounds)
    #print(fullmaxbounds)
    #print(noiselevels)
    #print(squash_levels)
    #print(np.median(squash_levels))
    #print(1e6*np.array(T))
    #print(hp[1][0:20])

    return [T, Dgx]


t2 = temp_path_list(path_list_temp, path_high_pressure_nofb, file_high_pressure_nofb, path_calibration, pathno,
                    acceleration_plot)

plt.show()
