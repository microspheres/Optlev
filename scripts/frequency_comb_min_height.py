import numpy as np
import matplotlib.pyplot as plt


# frequency_list = [31., 41., 47., 67., 71., 73., 89.]
# frequency_list = [31. - 5., 41. -3., 47.+3., 67.-5., 71.-7., 73.+5., 89.-5.]

frequency_list = [41.0, 47.3, 70.7, 73.0]




lenght = 2**16

sampling_freq = 10000.

# close_to_zero = 0.2
# close_to_zero_mean = 0.022

def random_function(f,sf):
    a = 4
    best = []
    for i in range(100000):
        function = np.zeros(lenght)
        time = np.arange(0,lenght)/sf
        for freq in f:
            function += 1.0*np.sin(2.0*np.pi*freq*time + 2.0*np.pi*np.random.rand())
        m = np.max(np.abs(function))
        if m < a:
            a = m
            best = function
    function = best
    print np.max(function)
    return function/np.max(function), time


f,t = random_function(frequency_list,sampling_freq)

np.savetxt("waveform_zero_edge_frequency_comb.txt", f)

print f[0]
print f[lenght -1]
print np.mean(f)

plt.plot(t,f)
plt.show()
