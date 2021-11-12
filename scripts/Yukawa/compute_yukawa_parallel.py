import numpy as np
import force as fo
import multiprocessing as mp
import os

# X true calculates in x direction and Y true calculates in y direction. They can NOT be true at the same time.
X = True
Y = False


path_save = r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\Yukawa\10um_sphere_JAN_2020"

pi = np.pi

Sph_diam = 10.3e-6

Drop_rad = 45.0e-6

Drop_len = 100.0e-6

Drop_x0 = -1.0*Drop_len/2.0

dist = 1.0e-6

wall = 5.0e-6

center_center_distance = Drop_rad + Sph_diam/2. + wall + dist

rho_drop = 1900. - 800.

rho_sphe = 1800.

mass_sph = (4./3.)*pi*((Sph_diam/2.)**3)*rho_sphe

lam_array = 10**(np.arange(-6,-3.9, 0.1))

alpha = 1.0

print (lam_array)

if X:
    file_name = "sphere_dia_" + str(1.0e6*Sph_diam) + "um_dist_" + str(1.0e6*dist) + "um_drop_rad_" + str(1.0e6*Drop_rad)+ "um_wall_thickness_" + str(1.0e6*wall) + "um_drop_len_" + str(1.0e6*Drop_len) + "um_X_direction.npy"
if Y:
    file_name = "sphere_dia_" + str(1.0e6*Sph_diam) + "um_dist_" + str(1.0e6*dist) + "um_drop_rad_" + str(1.0e6*Drop_rad)+ "um_wall_thickness_" + str(1.0e6*wall) + "um_drop_len_" + str(1.0e6*Drop_len) + "um_Y_direction.npy"

pathname = os.path.join(path_save, file_name)


def calculate(i):

    if X:
        a = fo.Forcex(rho_drop, rho_sphe, alpha, i, Sph_diam, Drop_rad, Drop_len, center_center_distance, Drop_x0)[0]/((9.8e-9)*mass_sph)
    if Y:
        a = fo.Forcey(rho_drop, rho_sphe, alpha, i, Sph_diam, Drop_rad, Drop_len, center_center_distance, Drop_x0)[0]/((9.8e-9)*mass_sph)

    return a
    
if __name__ == '__main__':
    
    mp.freeze_support()
    pool = mp.Pool(processes=8)
    
    k = pool.map(calculate, lam_array)
    
    np.save(pathname, [lam_array, k])

print pathname
