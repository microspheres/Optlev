import numpy as np
import matplotlib.pyplot as plt
import os
import glob



drop_len = 50 # um

# there is only one file per folder. see compute_yukawa_parallel_time.py

folder_list = ["/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/Different_drop_positions/lambda_100.0um",
               "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/Different_drop_positions/lambda_50.0um",
               "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/Different_drop_positions/lambda_20.0um",
               "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/Different_drop_positions/lambda_10.0um",
               "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/Different_drop_positions/lambda_1.0um",]

plt.figure()
for i in folder_list:
    
    file = glob.glob(os.path.join(i,'*.npy'))
    
    acc = []
    pos = []
    # there is only one file per folder
    L = np.load(file[0])
    for j in L:
        a = j[1]
        p = j[0]
        acc.append(a)
        pos.append(p)
    plt.semilogy(np.array(pos)*1e5 + 2.5, -1.*np.array(acc), label = "$\lambda$ = " + str(file[0][108:-83]))
            

plt.xlabel("Position inside capillary ($\mu$m)")
plt.ylabel("Acceleration in nano-$g$ for "+r"$\alpha$" +" = 1")
plt.grid()
plt.legend()
plt.title("drop len =" + str(drop_len) + "$\mu$m" + ", distance = 15 $\mu$m")
plt.ylim(1e-15, 1e-2)
plt.tight_layout(pad = 0)
plt.show()