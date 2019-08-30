import numpy as np
import matplotlib.pyplot as plt
import os

plot_current_limit = True

time_1week = 24.*3600.*7

acc22 = 500./np.sqrt(time_1week) # in nano-g

acc15 = 1000./np.sqrt(time_1week) # in nano-g

hollow = False


plot = True

#path_load = "/Users/fernandomonteiro/Desktop/Python/Force_yukawa/Computed_files_analytic/20180803"
#
#file_list = ["sphere_dia_15.0um_dist_5.0um_drop_rad_40.0um_drop_len_50.0um.npy",
#             "sphere_dia_15.0um_dist_5.0um_drop_rad_25.0um_drop_len_50.0um.npy",
#             "sphere_dia_22.0um_dist_5.0um_drop_rad_25.0um_drop_len_50.0um.npy",
#             "sphere_dia_22.0um_dist_5.0um_drop_rad_40.0um_drop_len_50.0um.npy"]

path_load = r"C:\Users\yalem\OneDrive\Desktop\CL_droplets\Yukawa"

file_list = [r"sphere_dia_15.0um_dist_2.0um_drop_rad_45.0um_wall_thickness_5.0um_drop_len_200.0um.npy",
             r"sphere_dia_22.0um_dist_2.0um_drop_rad_45.0um_wall_thickness_5.0um_drop_len_200.0um.npy"]
             #r"sphere_dia_15.0um_dist_2.0um_drop_rad_25.0um_wall_thickness_3.0um_drop_len_200.0um.npy"]


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
            
    return [Lambda, alpha]


print (alpha_Lambda(file_list)[0][0])
print (alpha_Lambda(file_list)[1][0])

al = alpha_Lambda(file_list)


if plot:
    plt.figure()
    for i in range(len(file_list)):
        plt.loglog(al[0][i], al[1][i], label = str(file_list[i][69:-21]) )


    plt.title("Data of Phys. Rev. A 96, 063841 - 1 week integration time")
    plt.xlabel("$\lambda$ (m)")
    plt.ylabel(r"$\mid\alpha\mid$")


if plot_current_limit:

    LL = np.genfromtxt(os.path.join(r"C:\Users\yalem\OneDrive\Desktop\CL_droplets\Calculations", "master_new.txt"), delimiter = " ")


    #plt.figure()
    #plt.loglog(LL[:,0], LL[:,1], "k-")
    plt.plot(LL[:,0], LL[:,1],"k--")
    plt.fill_between(LL[:,0], LL[:,1], 1.0e20*np.ones(len(LL[:,1])), alpha = 0.2)
    plt.xlim(2.0e-6, 2.0e-5)
    plt.ylim(100., 1.0e7)



plt.grid()
plt.legend()
plt.tight_layout(pad = 0)
plt.show()
