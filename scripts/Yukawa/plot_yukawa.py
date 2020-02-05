import numpy as np
import matplotlib.pyplot as plt
import os

plot_current_limit = True

time_1week = 24.*3600.

acc22 = 400./np.sqrt(time_1week) # in nano-g

acc15 = 1000./np.sqrt(time_1week) # in nano-g

acc10 = 95./np.sqrt(time_1week) # in nano-g

hollow = False

plot = True

#path_load = "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/20180803"
#
#file_list = ["sphere_dia_15.0um_dist_5.0um_drop_rad_40.0um_drop_len_50.0um.npy",
#             "sphere_dia_15.0um_dist_5.0um_drop_rad_25.0um_drop_len_50.0um.npy",
#             "sphere_dia_22.0um_dist_5.0um_drop_rad_25.0um_drop_len_50.0um.npy",
#             "sphere_dia_22.0um_dist_5.0um_drop_rad_40.0um_drop_len_50.0um.npy"]

path_load = r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa\results_fixed_droplet_position"

path_load = r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa\10um_sphere_JAN_2020"



file_list = [r"sphere_dia_22.0um_dist_2.0um_drop_rad_70.0um_wall_thickness_10.0um_drop_len_50.0um_X_direction.npy", r"sphere_dia_22.0um_dist_4.0um_drop_rad_70.0um_wall_thickness_10.0um_drop_len_50.0um_X_direction.npy"]

file_list = [r"sphere_dia_10.3um_dist_1.0um_drop_rad_45.0um_wall_thickness_5.0um_drop_len_100.0um_X_direction.npy" ,r"sphere_dia_10.3um_dist_5.0um_drop_rad_45.0um_wall_thickness_5.0um_drop_len_100.0um_X_direction.npy", r"sphere_dia_10.3um_dist_10.0um_drop_rad_45.0um_wall_thickness_5.0um_drop_len_100.0um_X_direction.npy",]


FL = []
for i in file_list:
    
    fl = os.path.join(path_load, i)
    
    FL.append(fl)
    
file_list = FL

if hollow:

    path_load2 = "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic_hollow/20181124"

    file_list2 = ["sphere_dia_22.0um_dist_5.0um_drop_rad_40.0um_drop_len_50.0um.npy"]


    FL2 = []
    for i in file_list2:
    
        fl = os.path.join(path_load2, i)
    
        FL2.append(fl)
    
    file_list = file_list+FL2




def alpha_Lambda(file_list):
    
    alpha = []
    Lambda = []
    
    for i in file_list:
        
        if "sphere_dia_22.0um" in i:
            
            k = np.load(i)
            
            a = -acc22/k[1]
            
            alpha.append(a)
            Lambda.append(k[0])
        
        if "sphere_dia_15.0um" in i:
            
            k = np.load(i)
            
            a = -acc15/k[1]
            
            alpha.append(a)
            Lambda.append(k[0])

        if "sphere_dia_10.3um" in i:
            
            k = np.load(i)
            
            a = -acc10/k[1]
            
            alpha.append(a)
            Lambda.append(k[0])
            
    return [Lambda, alpha]


#print (alpha_Lambda(file_list)[0][0])
#print (alpha_Lambda(file_list)[1][0])

al = alpha_Lambda(file_list)

plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots()
if plot:
    for i in range(len(file_list)):
        ax.loglog(1e6*al[0][i], al[1][i])#, label = str(file_list[i][69:-21]) )
        ax.fill_between(1e6*al[0][i], al[1][i], 1.0e20*np.ones(len(al[1][i])), alpha = i/(2*len(file_list))+0.25, color = [0.7, 0.7, 0.7])

if plot_current_limit:

    LL = np.genfromtxt(os.path.join(r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa", "master_new.txt"), delimiter = " ")


    # plt.figure()
    # plt.loglog(LL[:,0], LL[:,1], "k-")
    ax.plot(1e6*LL[:,0], LL[:,1],"k--")
    ax.fill_between(1e6*LL[:,0], LL[:,1], 1.0e20*np.ones(len(LL[:,1])), color = [0.5, 0.75, 1])


ax.set_xticks([1, 10, 20])

#ax.set_xticklabels([])
# plt.grid()
# plt.legend()
plt.xlim(1.0, 20)
plt.ylim(100., 1.0e7)
ax.set_xlabel("$\lambda$ [$\mu$m]")
ax.set_ylabel(r"$\mid\alpha\mid$")
plt.tight_layout(pad = 0)
fig.set_size_inches(4,4.5)
plt.show()
