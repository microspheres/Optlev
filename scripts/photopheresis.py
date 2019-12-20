import numpy, h5py, matplotlib
import matplotlib.pyplot as plt
import scipy.signal as sp
import numpy as np
import scipy.optimize as opt
import glob

colors = ['#1f78b4', '#e66101', '#33a02c', '#984ea3', '#F27781', '#18298C', '#04BF8A', '#F2CF1D', '#F29F05', '#7155D9', '#8D07F6', '#9E91F2', '#F29B9B', '#F25764', '#6FB7BF', '#B6ECF2', '#5D1314', '#B3640F']

pi = np.pi

def P0(r, eta, Rgas, Tgas, MolecularMass, acomo):
    acomo = np.abs(acomo)
    a = 3.*eta*np.sqrt(Rgas*Tgas/(MolecularMass*acomo))/r
    return a

def force_constant_grad(p, grad, Rgas, r, eta, Tgas, MolecularMass, acomo): # the gradient is constant in pressure and is prop to co2 power

    p0 = P0(r, eta, Rgas, Tgas, MolecularMass, acomo)
    print p0

    n = 1.*pi*eta*(r)*np.sqrt(acomo*Rgas/(MolecularMass*Tgas))*grad # there is a typo in wenqgian's formular. It is linear in the radius
    d = 1.*p/p0 + 1.*p0/p

    return n/d

def force_grad_vs_p(p, grad, Rgas, r, eta, Tgas, MolecularMass, acomo, cut):

    p0 = P0(r, eta, Rgas, Tgas, MolecularMass, acomo)
    print p0

    result = np.ones(len(p))
    
    for i in range(len(p)):
        if p[i] < cut: # cut is in Pascal
            n = 1.*pi*eta*(r)*np.sqrt(acomo*Rgas/(MolecularMass*Tgas))*grad # there is a typo in wenqgian's formular. It is linear in the radius
            d = 1.*p[i]/p0 + 1.*p0/p[i]
            result[i] = result[i]*n/d 
        else:
            n = 1.*pi*eta*(r)*np.sqrt(acomo*Rgas/(MolecularMass*Tgas))*grad/(p[i]/cut)
            d = 1.*p[i]/p0 + 1.*p0/p[i]
            result[i] = result[i]*n/d     

    return result

def Intensity_to_loose_constant_grad(p, Rgas, r, eta, Tgas, MolecularMass, constant, acomo): # the gradient is constant in pressure and is prop to co2 power
    acomo = np.abs(acomo)
    p0 = P0(r, eta, Rgas, Tgas, MolecularMass, acomo)

    a = 1.*constant*(1.*p/p0 + 1.*p0/p)

    b = 1.*pi*eta*(r)*np.sqrt(acomo*Rgas/(MolecularMass*Tgas)) # there is a typo in wenqgian's formular. It is linear in the radius
    
    return a/b

def Intensity_to_loose_gradoverpress(p, Rgas, r, eta, Tgas, MolecularMass, constant, acomo, cut): # the gradient is constant in pressure and is prop to co2 power
    acomo = np.abs(acomo)
    p0 = P0(r, eta, Rgas, Tgas, MolecularMass, acomo)

    result = np.ones(len(p))

    for i in range(len(p)):
        if p[i] < cut: # cut is pressure in Pa
            n = 1*constant*(1.*p[i]/p0 + 1.*p0/p[i])
            d = 1.*pi*eta*(r)*np.sqrt(acomo*Rgas/(MolecularMass*Tgas)) # there is a typo in wenqgian's formular. It is linear in the radius
            result[i] = result[i]*n/d 
        else:
            n = p[i]*constant*(1.*p[i]/p0 + 1.*p0/p[i])/cut
            d = 1.*pi*eta*(r)*np.sqrt(acomo*Rgas/(MolecularMass*Tgas))
            result[i] = result[i]*n/d  
    
    
    return result

Press = np.logspace(-1, 5, 100) # in Pascal

plt.figure()
plt.loglog(Press/100, force_constant_grad(Press, 1e-2, 8.3, 7.5e-6, 1e-5, 300, 30e-3, 1))
plt.loglog(Press/100, force_grad_vs_p(Press, 1e-2, 8.3, 7.5e-6, 1e-5, 300, 30e-3, 1, 1))
plt.xlabel("Pressure mbar")



folder = r"C:\Users\yalem\GitHub\Documents\Optlev\scripts\wenqiang_power_to_loose"
file_list = glob.glob(folder+"\*.txt")

plt.figure()
x = []
y = []
for i in file_list:
    a = np.loadtxt(i)
    b = np.transpose(a)
    x = np.append(x,b[0])
    y = np.append(y, b[1]/25*1.4)

ind = np.argsort(x)

x = x[ind]
y = y[ind]
pmin_ind = np.where(x > 2)[0][0]

x = x*1e2 # it is in pascal


popt, pcov = opt.curve_fit(lambda press, k, acomo: Intensity_to_loose_constant_grad(press, 8.31, 7.5e-6, 1.8e-5, 300., 29e-3, k, acomo), x[pmin_ind:], y[pmin_ind:])
print popt
popt = [8.31, 7.5e-6, 1.8e-5, 300., 29e-3, popt[0], popt[1]]

popt2, pcov2 = opt.curve_fit(lambda press, k, acomo, cut: Intensity_to_loose_gradoverpress(press, 8.31, 7.5e-6, 1.8e-5, 300., 29e-3, k, acomo, cut), x[pmin_ind:], y[pmin_ind:])
print popt2
popt2 = [8.31, 7.5e-6, 1.8e-5, 300., 29e-3, popt2[0], popt2[1], popt2[2]]

chi2_model_Ashkin = np.sum( (y - Intensity_to_loose_constant_grad(x, *popt))**2 / Intensity_to_loose_constant_grad(x, *popt)  )

# print chi2_model_Ashkin

chi2_model_Geraci = np.sum( (y - Intensity_to_loose_gradoverpress(x, *popt2))**2 / Intensity_to_loose_gradoverpress(x, *popt2)  )

# print chi2_model_Geraci

plt.subplot(2,1,1)
plt.semilogx(x, y, "o")
plt.ylabel('laser power(mW/mm^2)')
plt.title('loosing power')
plt.plot(Press, Intensity_to_loose_constant_grad(Press, *popt))
plt.plot(Press, Intensity_to_loose_gradoverpress(Press, *popt2))
plt.ylim(0, 70)
plt.xlim(1, 1e5)
plt.subplot(2,1,2)
plt.semilogx(x, y - Intensity_to_loose_constant_grad(x, *popt), "o")
plt.plot(x, y - Intensity_to_loose_gradoverpress(x, *popt2), "x")
plt.xlim(1, 1e5)
plt.ylim(-15, 15)
plt.xlabel('pressure(Pa)')
# plt.show()


plt.figure()
x = []
y = []
for i in range(len(file_list)):
    a = np.loadtxt(file_list[i])
    b = np.transpose(a)
    x = b[0]
    y = b[1]/25*1.4

    ind = np.argsort(x)
    x = x[ind]
    y = y[ind]
    pmin_ind = np.where(x > 5)[0][0]
    x = x*1e2
    
    popt, pcov = opt.curve_fit(lambda press, k, acomo: Intensity_to_loose_constant_grad(press, 8.31, 7.5e-6, 1.8e-5, 300., 29e-3, k, acomo), x[pmin_ind:], y[pmin_ind:])
    popt = [8.31, 7.5e-6, 1.8e-5, 300., 29e-3, popt[0], popt[1]]

    popt2, pcov2 = opt.curve_fit(lambda press, k, acomo, cut: Intensity_to_loose_gradoverpress(press, 8.31, 7.5e-6, 1.8e-5, 300., 29e-3, k, acomo, cut), x[pmin_ind:], y[pmin_ind:], p0 = [ 4.63075646e-12, -6.13007360e-03,  1.10704947e+03])
    popt2 = [8.31, 7.5e-6, 1.8e-5, 300., 29e-3, popt2[0], popt2[1], popt2[2]]

    chi2_model_Ashkin = np.sum( (y - Intensity_to_loose_constant_grad(x, *popt))**2 / Intensity_to_loose_constant_grad(x, *popt)  )
    print chi2_model_Ashkin

    chi2_model_Geraci = np.sum( (y - Intensity_to_loose_gradoverpress(x, *popt2))**2 / Intensity_to_loose_gradoverpress(x, *popt2)  )
    print chi2_model_Geraci
    
    plt.semilogx(x, y, "o", color = colors[i])
    plt.plot(Press, Intensity_to_loose_constant_grad(Press, *popt), color = colors[i], linestyle = "--")
    plt.plot(Press, Intensity_to_loose_gradoverpress(Press, *popt2), color = colors[i])


plt.xlabel('pressure(Pa)')
plt.ylabel('laser power(mW/mm^2)')
plt.title('loosing power')
plt.ylim(0, 70)
plt.xlim(1, 1e5)

plt.show()

