import numpy as np
import force as fo
import multiprocessing as mp
import os


# lambda and alpha are fixed but the drop initial positon Drop_x0_array changes

# there is only one saved file per folder

path_save = r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa\results_Variable_droplet_position"

pi = np.pi

Sph_diam = 2.2e-5

Drop_rad = 4.5e-5

Drop_len = 5.0e-5

Drop_x0_array = np.arange(-30, 30, 0.5)*1.0e-5

dist = 2.e-6

center_center_distance = Drop_rad + Sph_diam/2. + 5.0e-6 + dist

rho_drop = 2000. - 800.

rho_sphe = 1800.

mass_sph = (4./3.)*pi*((Sph_diam/2.)**3)*rho_sphe

lam = 1.0e-5

alpha = 1.0

path_save = os.path.join(path_save, "lambda_" + str(lam*1.0e6) + "um")

if not os.path.exists(path_save):
    os.makedirs(path_save)


def calculate(i):
    
    ax = fo.Forcex(rho_drop, rho_sphe, alpha, lam, Sph_diam, Drop_rad, Drop_len, center_center_distance, i)[0]/((9.8e-9)*mass_sph)
    ay = fo.Forcey(rho_drop, rho_sphe, alpha, lam, Sph_diam, Drop_rad, Drop_len, center_center_distance, i)[0]/((9.8e-9)*mass_sph)

    return [i, ax, ay, dist, Drop_len, lam]



if __name__ == '__main__':
    
    mp.freeze_support()

    pool = mp.Pool(processes=8)
    
    k = pool.map(calculate, Drop_x0_array)


    file_name = "sphere_dia_" + str(1.0e6*Sph_diam) + "um_dist_" + str(1.0e6*dist) + "um_drop_rad_" + str(1.0e6*Drop_rad) + "um_drop_len_" + str(1.0e6*Drop_len) + "several_positions" +"um.npy"
    pathname = os.path.join(path_save, file_name)
    
    np.save(pathname, k)
    
    print ("finished")










# version without multiprocessing
#def calculate(i):
#    
#    a = fo.Force(rho_drop, rho_sphe, alpha, lam, Sph_diam, Drop_rad, Drop_len, center_center_distance, i)[0]/((9.8e-9)*mass_sph)
#
#    return a
#    
#
#for i in Drop_x0_array:
#    
#    file_name = "sphere_dia_" + str(1.0e6*Sph_diam) + "um_dist_" + str(1.0e6*dist) + "um_drop_rad_" + str(1.0e6*Drop_rad) + "um_drop_len_" + str(1.0e6*Drop_len) + "um_drop_x0_" + str(1.0e6*i) + "um.npy"
#    pathname = os.path.join(path_save, file_name)
#        
#    k = calculate(i)
#    
#    np.save(pathname, [i, k])
#
#    print ("doing")
#    
#print ("finished")
