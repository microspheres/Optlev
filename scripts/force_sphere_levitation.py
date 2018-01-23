import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

# force due to the trap laser on the vertical direction.

Pi = np.pi

waist = 25.0*1e-6 # waist at the sphere

f = 25.0*1e-3 # 25.0 mm

l = 1064.*1e-9 # 1064 nm

w0 = 0.772*1e-3 # error 0.017*1e-3

w0err = 0.017*1e-3

def waist_focus_calculator(w0,l,f):
    aux = 1.0*(f*l/Pi)**2
    aux2 = 0.5*( - np.sqrt( w0**4 - 4.*aux ) + 1.*w0**2)
    return np.sqrt(aux2)

waist_focus = waist_focus_calculator(w0,l,f)

def sphere_height(waist, waist_focus):
    return ((Pi*waist_focus**2)/l)*np.sqrt((waist/waist_focus)**2 - 1)

height = sphere_height(waist, waist_focus)
print "height", height
print "waits at focus", waist_focus
print "NA", l/(Pi*waist_focus)
print "NA another way to calculate", w0/f
print "NA laser", l/(Pi*w0)
print "laser_waist", w0

n = 1.45 #1.45 at 1064 nm

c = 299792458.0

P_laser = 1. # watt




def P(beta, w0, f, P_laser):
    aux = (2.0*P_laser/(Pi*w0**2))*np.exp(-2*(f*beta/w0)**2)
    return aux

def theta(beta, Radius, height): # angle of incidence
    aux = 1.0*height*beta/Radius
    return aux

def thetaF(n,theta): # angle of refraction (small r of eqs 1,2 in the  paper)
    aux = np.arcsin(np.sin(theta)/n)
    return aux

def R(theta,n): # fresnel reflection
    aux1 = 1.0*(np.cos(theta) - n*np.cos(thetaF(n,theta)))
    aux2 = 1.0*(np.cos(theta) + n*np.cos(thetaF(n,theta)))
    aux = 1.0*np.abs(aux1/aux2)**2
    return aux

def T(theta,n): # fresnel trans
    return 1.0 - R(theta,n)

##### eq1

def Fs(beta ,n, height, Radius, w0, f, P_laser):
    aux1 = 1.0*n*P(beta, w0, f, P_laser)/c
    aux2 = 1.0
    aux3 = R(theta(beta, Radius, height),n)*np.cos(2.0*theta(beta, Radius, height))
    aux4n = 1.0*(T(theta(beta, Radius, height),n)**2)*(np.cos( 2.0*theta(beta, Radius, height) - 2.0*thetaF(n,theta(beta, Radius, height)) ) + R(theta(beta, Radius, height),n)*np.cos(2.0*theta(beta, Radius, height)))
    aux4d = 1.0 + R(theta(beta, Radius, height),n)**2 + 2.0*R(theta(beta, Radius, height),n)*np.cos(2.0*thetaF(n,theta(beta, Radius, height)))
    return aux1*(aux2 + aux3 + aux4n/aux4d)





# approx INTEGRAL Fs*(height**2)*beta*Dbeta from 0 to 0.005

def integral(Radius, w0):
    Dbeta = 0.00001
    beta_max = np.min([Radius/height, w0/f])
    beta = np.arange(0, beta_max, Dbeta)

    fs = 0.
    for i in beta:
        fs =+ Fs(i ,n, height, Radius, w0, f, P_laser)*i*(Dbeta)
    # print "force"
    # print fs

    # print "mass in ng"
    # print (fs/9.8)*1e12
    return fs

Ra = np.logspace(-6, -4, 100)
list1 = np.zeros([len(Ra),3])
for i,r in enumerate(Ra):
    aux = integral(r, w0)
    aux1 = 0#integral(r, w0 - w0err)
    aux2 = 0#integral(r, w0 + w0err)
    list1[i,:] = [aux, aux1, aux2]

Ma = (4./3.)*Pi*(Ra**3)*(1850)*(1e12) #ng
Ma1 = np.vstack([Ma, Ma, Ma]).T


Pa = (Ma1*(1e-12)*9.8/list1)*1e3 # levitation power mW

realdiameter = np.array([14.7, 14.4, 13.6, 16.2, 11.1, 1.026*5.7, 1.026*5.0, 1.026*5.0, 24.4, 22.6, 22.3, 33.4, 33.0])/1e6 # in m the 1.02 factor on the 5um spheres is due to its higher density: 1.026**3 = 2.0/1.85

realmass = 1850.*(4./3.)*Pi*((realdiameter)/2.0)**3 # in kg
realmass = realmass*1e12 # in ng

realpower = np.array([684., 777., 586., 811., 425, 71., 65., 68., 2105., 1820., 1827., 2682., 3075.])/1000. #mW
realpower = realpower*122.9*0.93 # see conversion table


dia_pixel = realdiameter/(0.359*1e-6)
Ddia_err_pixel_list = 1./dia_pixel # 1 pixel of error
Ddia_err_pixel = 0.032 + Ddia_err_pixel_list

realmasserr = realmass*3.0*Ddia_err_pixel

realmasserr = realmasserr + 0.1*realmasserr # 10% density systematics


##################################################################################### Unrelated

m = np.array([11.8, 10.2, 2.9, 2.7, 2.2, 1.3, 0.13])


Ddia_err_list = np.array([1./64., 1./61., 1./40., 1./39., 1./37., 1./31., 1./14.]) # this is systematic

Ddia_err = 0.032 + Ddia_err_list

Dmerr = 3*Ddia_err

merr = m*3.*Ddia_err

merr = merr + 0.1*merr # density uncertanty systematics

acc = ((1.0*10**-5)*np.array([0.38, 0.65, 0.88, 1.20, 1.18, 2.5, 3.3])/9.8)*1.e6 # in ug

accerr = acc*np.sqrt((0.1/2.1)**2 + (Dmerr)**2)

accerr = accerr + 0.1*accerr # due to 10% of density error

print Ddia_err*0.359/Ddia_err_list
print Ddia_err
print 0.359/Ddia_err_list


yaxis = [0.1,1,10]

plt.figure()

g = gs.GridSpec(2,1, hspace = 0.07)

plt.subplot(g[0])
plt.loglog(Ma, Pa[:,0], label = "Model", color = "k")
# plt.fill_between(Ma, Pa[:,1], Pa[:,2])
plt.errorbar(realmass, realpower, yerr = 0.02*realpower, xerr = realmasserr, fmt='ro', markersize = 4)
# plt.legend(loc = "", fontsize = 17, frameon = False)
plt.ylabel("Laser Power [mW]", fontsize = 17)
# plt.xlabel("Mass [ng]", fontsize = 17)
plt.xlim(0.08, 40)
plt.ylim(5, 1000)
plt.yticks([10., 100., 1000.] ,fontsize=16)
plt.gca().set_yticklabels([10, 100, 1000])
plt.gca().set_xticklabels([])
plt.xticks(fontsize=16)
plt.grid()





plt.subplot(g[1])

plt.errorbar(m, acc, xerr=merr, yerr = accerr, fmt='ro', markersize = 4)
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Mass [ng]", fontsize = 17)
plt.ylabel("$\sqrt{S_a}$ [$\mu g$/$\sqrt{Hz}$]", fontsize = 17)
plt.yticks(yaxis, fontsize=16)
plt.xticks(yaxis, fontsize=16)
plt.gca().set_yticklabels(yaxis)
plt.gca().set_xticklabels([0.1, 1.0, 10])
plt.xticks(fontsize=16)
# plt.grid(which = "minor")
plt.xlim(0.08, 40)
plt.ylim(0.1,10)
plt.grid(which = "major")
# plt.subplots_adjust(bottom = 0.15))
plt.legend(fontsize=16)
plt.gcf().set_size_inches(6.4,5)
# plt.tight_layout()
plt.subplots_adjust(right = 0.99, top = 0.98, left = 0.15, bottom = 0.12)
plt.show()
