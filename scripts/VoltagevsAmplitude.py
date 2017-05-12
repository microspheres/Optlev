import numpy as np
from force_calibration_charge import getdata
path = r"/data/20170511/bead2_15um_QWP/new_sensor_feedback/charge7_piezo_56.9_74.9_75.4"
conversion = 4.1000e-13

def getACAmplitudeGraphs(file_list):
    """output AC voltages and corresponding amplitudes at both omega and 2 omega"""
    N = len(file_list)
    constant = conversion/N
    x = {} # input only numpy arrays as values
    dx = {} # input only numpy arrays as values
    voltageCount = {} # input integers that count how many times an AC voltage value has shown up
    for index in range(N):
        f = file_list[index]
        a = getdata(f)
        i = f.rfind("synth")+5
        j = f.rfind("mV")
        k = f.rfind("mV",0,j)
        ACvoltage = float(f[i:k])/1000.
        if ACvoltage in x:
            voltageCount[ACvoltage] += 1
            x[ACvoltage] += np.sqrt(a[1])
            dx[ACvoltage] += np.sqrt(a[2])
        else:
            voltageCount[ACvoltage] = 1
            x[ACvoltage] = np.sqrt(a[1])
            dx[ACvoltage] = np.sqrt(a[2])
    ACvoltages = x.keys()
    N1 = len(ACvoltages)
    omegaAmplitudes = range(N1)
    twoOmegaAmplitudes = range(N1)
    """Now insert the amplitude for the requisite frequencies"""
    for index in range(N1):
        i = np.argmax(dx[ACvoltages[index]])
        omegaAmplitudes[index] = constant*x[i]
        twoOmegaAmplitudes[index] = constant*x[2*i]
    return ACvoltages, omegaAmplitudes, twoOmegaAmplitudes
