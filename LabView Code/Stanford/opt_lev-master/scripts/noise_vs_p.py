import numpy as np
import bead_util as bu
import matplotlib.pyplot as plt
import glob, re

reprocess_file = True

path = r"C:\Data\20150823\Bead2\turbo_spin_down3"

def sort_fun(s):
    idx = re.findall("_\d+.h5", s)[0]
    return int( idx[1:-3] )

if reprocess_file:

    init_list = glob.glob(path + "\*.h5")
    files = sorted(init_list, key = sort_fun)
    dats = []
    for f in files:
        try:    
                cfile = f
                amp, bp, pcov, press = bu.fit_spec(cfile, [10., 200.] )
               
        except:
                print "uninformative error message"
                continue
        dat = np.hstack([amp, bp, press])
        dats.append(dat)


dats = np.array(dats)

p = dats[:, -1]
hi_pts = p < 1e-7
p[ hi_pts ] = dats[hi_pts,-2]

plt.figure()
plt.loglog(p, dats[:, 0],'-.')
plt.loglog(p, np.sqrt(dats[:, 1]),'-.')
#plt.plot(dats[:, -2], dats[:, 3])
plt.show()
