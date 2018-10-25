import numpy as np
import matplotlib.pyplot as plt


# frequency_list = [31., 41., 47., 67., 71., 73., 89.]
# frequency_list = [31. - 5., 41. -3., 47.+3., 67.-5., 71.-7., 73.+5., 89.-5.]

frequency_list = [47.3,]

AC_to_DC = 3.75/6.0


length = 50000

square_length = 12500

sampling_freq = 10000.

close_to_zero = 18
close_to_zero_mean = 2

def random_function(f,sf):
    a = 0
    while (a == 0):
        function = np.zeros(length)
        time = np.arange(0,length)/sf
        for freq in f:
            function += 1.0*np.sin(2.0*np.pi*freq*time + 0*2.0*np.pi*np.random.rand())
        if np.abs(function[0]) < close_to_zero and np.abs(function[length-1]) < close_to_zero and np.abs(np.mean(function)) < close_to_zero_mean:
            a = 1
    return function/np.max(function), time


pos_offset = np.ones(square_length)
neg_offset = -1.0*np.ones(square_length)


square_c = np.hstack([pos_offset,neg_offset])
square = []

while len(square) < length:
    square = np.hstack([square,square_c])
    

f,t = random_function(frequency_list,sampling_freq)

f = AC_to_DC*f + square

f = f/np.max(f)


np.savetxt("waveform_zero_edge_square_and_47_3Hz.txt", f)

print f[0]
print f[length -1]
print np.mean(f)

plt.plot(t,f)
plt.show()
