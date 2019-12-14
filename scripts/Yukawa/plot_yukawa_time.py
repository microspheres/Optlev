import numpy as np
import matplotlib.pyplot as plt
import os
import glob



# there is only one file per folder. see compute_yukawa_parallel_time.py

folder_list = [r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa\results_Variable_droplet_position\lambda_100.0um", r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa\results_Variable_droplet_position\lambda_10.0um"]

drop_speed = 1. # m/s

Aaccx = []
Aaccy = []
Apos = []

plt.figure()
for i in folder_list:
    
    file = glob.glob(os.path.join(i,'*.npy'))
    
    accx = []
    accy = []
    pos = []
    time = []
    # there is only one file per folder
    L = np.load(file[0])
    for j in L:
        ax = j[1]
        ay = j[2]
        p = j[0]
        dist = j[3]
        drop_len = j[4]
        lam = j[5]
        accx.append(ax)
        accy.append(ay)
        pos.append(p)
        time.append(p/drop_speed)
        
    plt.plot(np.array(pos)*1e5 + 2.5, -1.*np.array(accx), label = "X direction $\lambda$ = " + str(1e6 * lam) + " $\mu$m")
    plt.plot(np.array(pos)*1e5 + 2.5, -1.*np.array(accy), label = "Y direction $\lambda$ = " + str(1e6 * lam) + " $\mu$m")
    Aaccx.append(accx)
    Aaccy.append(accy)
    Apos.append(pos)
    
            

plt.xlabel("Position inside capillary ($\mu$m)")
plt.ylabel("Acceleration in nano-$g$ for "+r"$\alpha$" +" = 1")
plt.grid()
plt.legend()
distance = ", distance =" +  str(1e6 * dist) + " $\mu$m"
plt.title("drop len =" + str(1e6 * drop_len) + "$\mu$m" + distance)
plt.ylim(-3e-4, 8e-4)
plt.tight_layout(pad = 0)
plt.show()
