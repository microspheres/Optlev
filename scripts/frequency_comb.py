import numpy as np
import matplotlib.pyplot as plt


frequency_list = [0.001]
# frequency_list = [31. - 5., 41. -3., 47.+3., 67.-5., 71.-7., 73.+5., 89.-5.]

# frequency_list = [11.1, 33.3, 99.9, 133.2, 166.5, 199.8, 233.1, 266.4, 299.7, 399.6, 499.5, 666.0, 832.5, 1098.9, 1665.0, 1931.4, 2231.1]

# big_list = np.logspace(1.5, 3.5, 20)


# lenght = 2**16

sampling_freq = 10000.

# frequency_list = [8.,32., 40., 48., 56., 64., 72., 80.,96., 128., 168., 200., 208., 256.,320., 384., 512., 640., 768., 1024., 1280., 1632., 2048., 2560., 3072.]
lenght = 100.*(int(sampling_freq/frequency_list[0]))

close_to_zero = 0.1
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

def fixed_function(f,sf):
    function = np.zeros(lenght)[:-1]
    time = np.linspace(0,1,lenght)[:-1]
    for freq in f:
        function += 1.0*np.sin(2.0*np.pi*freq*time + 2.0*np.pi*np.random.rand())
        
    return function/np.max(function), time


def cons_forDC_function(f,sf):
    function = np.zeros(lenght)[:-1]
    time = np.linspace(0,1,lenght)[:-1]
    for freq in f:
        function += 1.0*np.sin(2.0*np.pi*freq*time) + (np.sin(2.0*np.pi*freq*time)-1)*(np.sin(50*2.0*np.pi*freq*time)-1) - 1
    
    return function/np.max(function), time



#whitenoise
def wn(sf):
    a = 0
    while (a == 0):
        function = np.zeros(lenght)
        time = np.arange(0,lenght)/sf
        for i in range(len(function)):
            function[i] = np.random.normal(0, 1)

        if np.abs(function[0]) < close_to_zero and np.abs(function[lenght-1]) < close_to_zero and np.abs(np.mean(function)) < close_to_zero_mean:
            a = 1
    return function/np.max(function), time

#kick
def kick(sf):
    a = 0
    while (a == 0):
        function = np.zeros(lenght)
        time = np.arange(0,lenght)/sf
        for i in range(len(function)):
            if i < 1000:
                function[i] = 1
            else:
                0

            a = 1
    return function/np.max(function), time


f,t = fixed_function(frequency_list,sampling_freq)
# f,t = kick(sampling_freq)

np.savetxt(r"C:\Users\UsphereLab\Documents\GitHub\Optlev\scripts\freq_comb\freq_0.001Hz.txt", f)
print len(f)

print f[0]
print f[-1]
print np.mean(f)

plt.plot(t,f)
plt.show()
