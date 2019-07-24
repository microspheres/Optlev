import numpy as np

hbar = 1.054571800 * 10 ** -34  # J s
speed_of_light = 299792458  # m/s
wavelength = 1064 * 10 ** -9  # m
energy_quantum = 2 * np.pi * hbar * speed_of_light / wavelength  # J


def jones_matrix_linear(phi, eta, theta):
    arb_11 = np.exp(1j * eta / 2.) * (np.cos(theta) ** 2) + np.exp(-1j * eta / 2.) * (np.sin(theta) ** 2)
    arb_12 = -1j * np.exp(-1j * phi) * np.sin(eta / 2.) * np.sin(2 * theta)
    arb_21 = -1j * np.exp(1j * phi) * np.sin(eta / 2.) * np.sin(2 * theta)
    arb_22 = np.exp(-1j * eta / 2.) * (np.cos(theta) ** 2) + np.exp(1j * eta / 2.) * (np.sin(theta) ** 2)
    return np.matrix([[arb_11, arb_12], [arb_21, arb_22]])


def change_of_basis_matrix_circ_to_lin():
    return np.sqrt(0.5) * np.matrix([[1, 1], [-1j, 1j]])


def jones_matrix_circular(phi, eta, theta):
    """ M_c = P_l_c M_l P_c_l """
    j_l = jones_matrix_linear(phi, eta, theta)
    c_l = change_of_basis_matrix_circ_to_lin()
    l_c = c_l.getH()
    return np.matmul(np.matmul(l_c, j_l), c_l)
    # return np.matmul(np.matmul(c_l, j_l), l_c)


def jones_calculus_change_in_momentum(init_rhp, phi, eta, theta):
    # init_rhp is the amount of right-handed light we have between 0 and 1
    j_c = jones_matrix_circular(phi, eta, theta)
    l_i = np.matrix([[np.sqrt(init_rhp)], [np.sqrt(1 - init_rhp)]])
    l_f = np.matmul(j_c, l_i)
    initial_momentum = hbar * (np.abs(l_i[1]) ** 2 - np.abs(l_i[0]) ** 2)
    final_momentum = hbar * (np.abs(l_f[1]) ** 2 - np.abs(l_f[0]) ** 2)
    return final_momentum - initial_momentum


def calculate_torque(delta_momentum, power):
    # power is in Watts (= J/s)
    photons_per_second = power / energy_quantum
    torque = photons_per_second * delta_momentum  # J = N m
    return torque  # N m


def jones_calculus_torque(init_rhp, power, phi, eta, theta):
    # init_rhp is the amount of right-handed light we have between 0 and 1
    # power is in Watts (= J/s)
    delta_momentum = jones_calculus_change_in_momentum(init_rhp, phi, eta, theta)
    return calculate_torque(delta_momentum, power)
