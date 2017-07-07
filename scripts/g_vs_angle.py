import numpy as np
import matplotlib.pyplot as plt
from VoltagevsAmplitude import getdata
from dipole_fit_scale import get_g0
import glob

path = r'C:\data\20170622\bead4_15um_QWP\dipole3_Y'
conversion = 3.7139625927e-12 # N/V and 10 times the previous one because the gain is 10 times higher

def get_thetaY_ACvoltage_twoOmegaAmplitudes(file_list, make_plots = False):
    """ returns the values in the function name for each file in file_list """
    N = len(file_list)
    ACvoltageValues = np.zeros(N)
    thetaYvalues = np.zeros(N)
    twoOmegaAmplitudes = np.zeros(N)
    if make_plots:
        plt.figure()
    for index in range(N):
        """ First get the input data """
        f = file_list[index]
        i = f.rfind("synth")+5
        j = f.rfind("mV")
        k = f.rfind("mV",0,j)
        ACvoltageValues[index] = float(f[i:k])/1000. # V
        m = f.rfind("stage_tilt_") + 11
        n = f.rfind("thetaY")
        thetaYvalues[index] = float(f[m:n]) # steps, so unitless
        """ Now get the measured data """
        freqs, signal, drive = getdata(f) # Hz, V^2/Hz, V^2/Hz
        binF = freqs[2] - freqs[1] # Hz
        indexPicked = np.argmax(drive)
        twiceFreqIndex = 2*indexPicked + 1 ##### THIS IS NOT RELIABLE SO CHECK EVERY TIME
        psd = np.sqrt(signal) # V/sqrtHz
        if make_plots:
            plt.loglog(freqs, psd)
            plt.loglog(freqs[twiceFreqIndex], psd[twiceFreqIndex], 'x')
        twoOmegaAmplitudes[index] = conversion*psd[twiceFreqIndex]*np.sqrt(binF) # N
    # print 'from get_thetaY_ACvoltage_twoOmegaAmplitudes'
    # print thetaYvalues
    # print ACvoltageValues
    # print twoOmegaAmplitudes
    # print ''
    if make_plots:
        plt.xlabel('frequencies [Hz]')
        plt.ylabel('psd [V/sqrtHz]')
        plt.show(block = False)
    return zip(*sorted(zip(thetaYvalues, ACvoltageValues, twoOmegaAmplitudes)))

def getGvsAngleGraphsY(path, make_plots = False):
    """ start with AC voltage value and theta Y value
        calculate g0 for each theta Y value from the AC voltage values
        plot g0 list vs theta Y list """
    file_list = glob.glob(path+"\*.h5")
    thetaYvalues, ACvoltageValues, twoOmegaAmplitudes = get_thetaY_ACvoltage_twoOmegaAmplitudes(file_list, make_plots)
    
    # get the length of each sub-list corresponding to a new g0 value
    # assuming each sub-list is of the same length (which it should be, based on the way we've done our 
    first = thetaYvalues[0]
    length = 0
    while thetaYvalues[length] == first:
        length += 1
    
    # split up ACvoltageValues and twoOmegaAmplitudes into the sub-lists for g0
    thetaY = np.unique(thetaYvalues)
    N = len(thetaY)
    voltageLists = np.array_split(ACvoltageValues, N)
    amplitudeLists = np.array_split(twoOmegaAmplitudes, N)
    ## if this was java, I'd put an assert(len(thetaY) == len(voltageLists) == len(amplitudeLists)) here

    g0 = np.zeros(N)
    for i in range(N):
        theta = thetaY[i]
        ACvoltages = voltageLists[i]
        forces = amplitudeLists[i]
        g0[i] = get_g0(ACvoltages, forces) # m^-1
        print g0[i]

    return thetaY, g0

def plotGvsAngleY(path, make_plots = False):
    thetaY, g0 = getGvsAngleGraphsY(path, make_plots)
    plt.figure()
    plt.plot(thetaY, g0)
    plt.xlabel('Steps in thetaY')
    plt.ylabel('Values of g0 [m^-1]')
    plt.title('g0 vs thetaY steps')
    plt.show()

plotGvsAngleY(path, make_plots = True)



file_list = glob.glob(path+"\*.h5")
# print 'File list: '
# print file_list
# print ' '
thetaYvalues, ACvoltageValues, twoOmegaAmplitudes = get_thetaY_ACvoltage_twoOmegaAmplitudes(file_list)
# print 'theta Y'
# print thetaYvalues
# print 'AC voltages'
# print ACvoltageValues
# print 'amp'
# print twoOmegaAmplitudes
# print ' '
    
# # get the length of each sub-list corresponding to a new g0 value
# # assuming each sub-list is of the same length (which it should be, based on the way we've done our 
first = thetaYvalues[0]
length = 0
while thetaYvalues[length] == first:
    length += 1

# # split up ACvoltageValues and twoOmegaAmplitudes into the sub-lists for g0
thetaY = np.unique(thetaYvalues)
# print 'theta Y abridged'
# print thetaY
# print 'also length = '
# print length
# print ' '
N = len(thetaY)
voltageLists = np.array_split(ACvoltageValues, N)
amplitudeLists = np.array_split(twoOmegaAmplitudes, N)
# ## if this was java, I'd put an assert(len(thetaY) == len(voltageLists) == len(amplitudeLists)) here

# print voltageLists
# print ACvoltageValues
# print amplitudeLists
# print twoOmegaAmplitudes

plt.figure()
plt.plot(voltageLists[0], amplitudeLists[0])
plt.show()

# # g0 = np.arange(N)
# # for i in range(N):
# #     theta = thetaY[i]
# #     ACvoltages = voltageLists[i]
# #     forces = amplitudeLists[i]
# #     g0[i] = get_g0(ACvoltages, forces) # m^-1
# #     print g0[i]
