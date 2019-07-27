
# coding: utf-8

# In[ ]:

import correlation_steps as corrsteps
import matplotlib.pyplot as plt
import numpy as np
get_ipython().magic(u'matplotlib notebook')

directory = "/data/20170726/bead1_15um_QWP/steps/"
calibration_path = directory + "calibration_1positive_47_3Hz/"
measurement_path = directory + "47_3_new/"


# In[ ]:

t, dc, corr = corrsteps.formData(measurement_path, calibration_path)


# In[ ]:

set(dc)


# In[ ]:

dc_corr_list_pos = []
dc_corr_list_neg = [0]
i = 0
for i in range(len(corr)):
    if dc[i] < 0:
        dc_corr_list_pos[-1]+=corr[i]
        dc_corr_list_neg.append(corr[i])
    else:
        dc_corr_list_pos.append(corr[i])
        dc_corr_list_neg[-1]+=corr[i]
dc_corr_list_neg = dc_corr_list_neg[1:]
time_steps = t[::2]
time_steps = np.array(time_steps) - time_steps[0]


# In[ ]:

from scipy.optimize import curve_fit
from scipy.stats import norm

def gaussian_distribution(x, A, u, sigma):
    return A * np.exp(-(x - u) ** 2 / (2 * sigma ** 2))


# In[ ]:

# plot the figure
pn, binp, patchs = plt.hist(dc_corr_list_pos, bins=20, normed=True, alpha=0.6, color='b')
mup, stdp = norm.fit(dc_corr_list_pos)
nn, binn, patchs = plt.hist(dc_corr_list_neg, bins=20, normed=True, alpha=0.6, color='y')
mun, stdn = norm.fit(dc_corr_list_neg)

plt.xlabel('Correlation [e]')
plt.ylabel('Occurances of correlation value')

plt.title('Blue = Normal Pairing; Yellow = Next Pairing')

plt.show()

mp = float(max(pn))
mn = float(max(nn))

if mup > 0:
    lboundp = [0.8*mp, 0.8*mup, 0.8*stdp]
    uboundp = [1.2*mp, 1.2*mup, 1.2*stdp]
else:
    lboundp = [0.8*mp, 1.2*mup, 0.8*stdp]
    uboundp = [1.2*mp, 0.8*mup, 1.2*stdp]

if mun > 0:
    lboundn = [0.8*mn, 0.8*mun, 0.8*stdn]
    uboundn = [1.2*mn, 1.2*mun, 1.2*stdn]
else:
    lboundn = [0.8*mn, 1.2*mun, 0.8*stdn]
    uboundn = [1.2*mn, 0.8*mun, 1.2*stdn]

plt.figure()

x = sorted(list(set(np.concatenate((binp, binn)))))
p = norm.pdf(x, mup, stdp)
plt.plot(x, p, 'g', linewidth=2)
n = norm.pdf(x, mun, stdn)
plt.plot(x, n, 'r', linewidth=2)

xp = (binp[1:] + binp[:-1]) / 2.
xn = (binn[1:] + binn[:-1]) / 2.

poptp, pcovp = curve_fit(gaussian_distribution, xp, pn, bounds=(lboundp, uboundp))
perrp = np.sqrt(np.diag(pcovp))
fitted_data_p = gaussian_distribution(xp, *poptp)
plt.plot(xp, fitted_data_p, 'b')
plt.errorbar(xp, pn, yerr=np.sqrt(pn), fmt='bo')

poptn, pcovn = curve_fit(gaussian_distribution, xn, nn, bounds=(lboundn, uboundn))
perrn = np.sqrt(np.diag(pcovn))
fitted_data_n = gaussian_distribution(xn, *poptn)
plt.plot(xn, fitted_data_n, 'y')
plt.errorbar(xn, nn, yerr=np.sqrt(nn), fmt='yo')

plt.xlabel('Correlation [e]')
plt.ylabel('Occurances of correlation value')

plt.title('Blue/Green = First; Yellow/Red = Second')

plt.show()


# In[ ]:

# print parameters
print 'fitting pos to gaussian gives:'
print '           mean = ', poptp[1], ' with error ', perrp[1]
print '    actual mean = ', mup
print ''
print '           standard deviation = ', poptp[2], ' with error ', perrp[2]
print '    actual standard deviation = ', stdp
print ''
print ''
print 'fitting neg to gaussian gives:'
print '           mean = ', poptn[1], ' with error ', perrn[1]
print '    actual mean = ', mun
print ''
print '           standard deviation = ', poptn[2], ' with error ', perrn[2]
print '    actual standard deviation = ', stdn

