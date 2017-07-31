import numpy as np
import matplotlib.pyplot as plt


# frequency_list = [31., 41., 47., 67., 71., 73., 89.]
# frequency_list = [31. - 5., 41. -3., 47.+3., 67.-5., 71.-7., 73.+5., 89.-5.]

frequency_list = [41.0, 47.3, 70.7, 73.0]




lenght = 2**16

sampling_freq = 10000.

close_to_zero = 0.2
close_to_zero_mean = 0.022

def random_function(f,sf):
    a = 0
    while (a == 0):
        function = np.zeros(lenght)
        time = np.arange(0,lenght)/sf
        for freq in f:
            function += 1.0*np.sin(2.0*np.pi*freq*time + 2.0*np.pi*np.random.rand())
        if np.abs(function[0]) < close_to_zero and np.abs(function[lenght-1]) < close_to_zero and np.abs(np.mean(function)) < close_to_zero_mean:
            a = 1
    return function/np.max(function), time


f,t = random_function(frequency_list,sampling_freq)

np.savetxt("waveform_zero_edge_frequency_comb.txt", f)

print f[0]
print f[lenght -1]
print np.mean(f)

plt.plot(t,f)
plt.show()
