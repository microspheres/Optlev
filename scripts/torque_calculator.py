import numpy as np

Pi = np.pi

def input(a,b,phi):
    f = np.array([a, b*np.exp(1j*phi)])
    N = np.sqrt(np.dot(f,f))
    return f/N

def HWP(theta):
    f = np.exp(-1j*Pi/2.)*np.matrix([[np.cos(2.*theta), np.sin(2.*theta)],[np.sin(2.*theta), -np.cos(2.*theta)]])
    return f

def QWP(theta):
    f = np.exp(-1j*Pi/4.)*np.matrix([[np.cos(theta)**2 + 1j*np.sin(theta)**2, (1.-1j)*np.sin(theta)*np.cos(theta)],[(1.-1j)*np.sin(theta)*np.cos(theta), 1j*np.cos(theta)**2 + np.sin(theta)**2]])
    return f

def ARB(theta, eta, phi): # arb WP. eta is the relative phase retardation betwen the fast and slow axis.
    f = np.matrix([[np.exp(1j*eta/2.)*np.cos(theta)**2 + np.exp(-1j*eta/2.)*np.sin(theta)**2, np.exp(-1j*phi)*(np.exp(1j*eta/2.) - np.exp(-1j*eta/2.))*np.sin(theta)*np.cos(theta)],[np.exp(1j*phi)*(np.exp(1j*eta/2.) - np.exp(-1j*eta/2.))*np.sin(theta)*np.cos(theta), np.exp(1j*eta/2.)*np.sin(theta)**2 + np.exp(-1j*eta/2.)*np.cos(theta)**2]])
    return f

def spin_op():
    f = np.matrix([[0, -1j],[1j, 0]])
    return f

print np.dot(HWP(1.3),input(1,1,0))
print np.dot(QWP(1.3),input(1,1,0))
print np.dot(ARB(1.3,-Pi/2,0),input(1,1,0))

# print "......."

# print np.dot( spin_op(), np.dot(HWP(1.3),input(1,1,0)).T)
# print np.real(np.dot( np.dot(HWP(1.3),input(1,1,0)) , np.dot( spin_op(), np.dot(HWP(1.3),input(1,1,0)).T)))[0,0]
