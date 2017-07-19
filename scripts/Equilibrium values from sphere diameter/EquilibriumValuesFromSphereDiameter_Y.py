import numpy as np
from scipy import stats

"""Universal Constants"""
n = 1.33  # index of diffraction
c = 2.998 * (10. ** 11.)  # milliwatts per Newtons
g = 9.807  # Newtons per kg; according to WolframAlpha
massDensity = 2. / (10. ** 3.)  # grams per cubic millimeter

def calculatedParameters(diameterInMicron, spacing, rmax, Smax):
    sphereRadius = diameterInMicron / 2000. # mm
    r_pts = np.arange(spacing, rmax * np.sqrt(np.log(10.)), spacing) # mm
    r = r_pts.reshape((len(r_pts), 1)) # mm
    S = np.arange(sphereRadius, Smax, spacing) # mm
    sphereVolume = 4. * np.pi * (sphereRadius ** 3.) / 3.  # mm cubed
    sphereMass = massDensity * sphereVolume / 1000.  # kg
    return sphereRadius, r, S, sphereMass

def get_R_Td_thetaR(theta):
    """ Returns the reflection coefficient
        and a function of the reflection and transmission coefficients """
    sinThetaR = np.sin(theta) / n # Snell's law
    thetaR = np.arcsin(sinThetaR)
    a = np.cos(theta)
    b = np.sqrt(1. - (np.sin(theta) / n) ** 2.)
    R = ((((n * a - b) / (n * a + b)) ** 2.) + (((a - n * b) / (a + n * b)) ** 2.)) / 2.
    T = 1.0 - R
    Td = (T ** 2.) / (1. + R ** 2. + 2. * R * np.cos(2. * thetaR))
    return R, Td, thetaR

def get_dQs_raw(theta, thetaR, R, Td):
    return 1. + R * np.cos(2. * theta) - Td * ((np.cos(2. * theta - 2. * thetaR) + R * np.cos(2. * theta)))

def get_dQg_raw(theta, thetaR, R, Td):
    return R * np.sin(2. * theta) - Td * ((np.sin(2. * theta - 2. * thetaR) + R * np.sin(2. * theta)))

def get_dQs(theta):
    R, Td, thetaR = get_R_Td_thetaR(theta)
    return get_dQs_raw(theta, thetaR, R, Td)

def get_dQg(theta):
    R, Td, thetaR = get_R_Td_thetaR(theta)
    return get_dQg_raw(theta, thetaR, R, Td)

def get_dQs_and_dQg(theta):
    R, Td, thetaR = get_R_Td_thetaR(theta)
    dQs = get_dQs_raw(theta, thetaR, R, Td)
    dQg = get_dQg_raw(theta, thetaR, R, Td)
    return dQs, dQg

def get_z_QofS(diameterInMicron, f, rmax, Smax, spacing):
    """ f, rmax, Smax, spacing, all in mm
        returns sphereMass in kg, S in mm, and unitless QofS """

    """Calculated parameters"""
    sphereRadius, r, S, sphereMass = calculatedParameters(diameterInMicron, spacing, rmax, Smax)

    phi = np.arctan(r / f)
    sinTheta = np.sin(phi) * S / sphereRadius
    isPhysical = np.abs(sinTheta) <= 1.
    theta = np.arcsin(sinTheta * isPhysical)

    # unitless force in terms of r and S; set dQs to 0 everywhere the sphere isn't.
    dQs = get_dQs(theta) * isPhysical * np.cos(phi)

    # integrate to get total force; jacobian is rdr; normalize with a gaussian
    thingToIntegrate = r * dQs * np.exp(-np.power(r, 2.) / (2. * np.power(rmax / 2., 2.)))
    QofS = 2 * np.pi * np.trapz(thingToIntegrate, x=r, axis=0)
    return sphereMass, S, QofS

def getTotalForceCurveFromQofS(P, sphereMass, QofS):
    constant = n * P / c  # Newtons
    downwardForce = sphereMass * g  # Newtons
    FofS = constant * QofS  # Newtons
    totalForce = FofS - downwardForce # Newtons
    return totalForce # Newtons

def postPowerValues(sphereMass, tolerance, spacing, QofS, P, S):
    """ f, rmax, Smax, spacing, sphereSize, and tolerance are all in mm
        P is in mW """
    totalForce = getTotalForceCurveFromQofS(P, sphereMass, QofS) # Newtons
    index = np.argmin(np.absolute(totalForce))
    equilibriumPosition = S[index] # mm
    # the number of points forwards and backwards used to calculate the trend
    tolerancePoints = int(np.ceil(tolerance / spacing))
    # take the points around this number to get the overall slope
    pointsAroundMass = totalForce[index - tolerancePoints:index + tolerancePoints] # Newtons
    xAroundMass = S[index - tolerancePoints:index + tolerancePoints]/1000. # meters
    # get the trendline to calculate the spring constant (Newtons per meter)
    slope, intercept, r_value, p_value, error = stats.linregress(xAroundMass, pointsAroundMass)
    springConstant = -slope # N/meter
    # calculate the resonant frequency from the spring constant
    resonantFrequency = 2.*np.pi*np.sqrt(springConstant / sphereMass)  # rad Hz
    return resonantFrequency, equilibriumPosition, error

def valuesFromParameters(f, rmax, Smax, spacing, diameterInMicron, P, tolerance):
    """f, rmax, Smax, spacing, sphereSize, and tolerance are all in mm
    P is in mW
    sphereSize is the sphere's diameter"""
    sphereMass, S, QofS = get_z_QofS(diameterInMicron, f, rmax, Smax, spacing)
    resonantFrequency, equilibriumPosition, error = postPowerValues(sphereMass, tolerance, spacing, QofS, P, S)
    return resonantFrequency, equilibriumPosition, error

def valuesFromSizePowerTolerance(diameterInMicron, PmW, tmm):
    """Plug experiment parameters into valuesFromParameters"""
    """Experiment parameters go here (make the numbers be floats)"""
    f = 25.  # mm; distance between the lens and the focus
    r = 1.  # mm; beam intensity is 1/e^2 at r = rmax
    S = 10.  # mm; highest the sphere will levetate as the distance from the focus
    sp = 1. / 1000.  # mm; this is how far apart our data points will be for r, S, and K
    resonantFrequency, equilibriumPosition, error = valuesFromParameters(f, r, S, sp, diameterInMicron, PmW, tmm)
    return resonantFrequency, equilibriumPosition, error

def valuesFromSphereSizeAndPower(diameterInMicron, PmW):
    """Plug experiment parameters into valuesFromSizePowerTolerance"""
    """Experiment parameters go here (make the numbers be floats)"""
    t = 0.1  # mm; how far to go on each side to calculate the trend for the spring constant
    resonantFrequency, equilibriumPosition, error = valuesFromSizePowerTolerance(diameterInMicron, PmW, t)
    return resonantFrequency, equilibriumPosition, error

def valuesFromSphereSizeAndPowers(diameterInMicron, rmax_mm, PmW_array):
    """Remake valuesFromParameters but for a power array;
       Also, beam intensity is 1/e^2 at r = rmax"""
    N = len(PmW_array)

    """Experiment parameters go here (make the numbers be floats)"""
    f = 25.  # mm; distance between the lens and the focus
    Smax = 10.  # mm; highest the sphere will levitate as the distance from the focus
    spacing = 1. / 1000.  # mm; this is how far apart our data points will be for r, S, and K
    t = 0.1  # mm; how far to go on each side to calculate the trend for the spring constant

    """Calculations"""
    sphereMass, S, QofS = get_z_QofS(diameterInMicron, f, rmax_mm, Smax, spacing)
    resonantFrequency = np.ones(N)
    equilibriumPosition = np.ones(N)
    error = np.ones(N)
    for i in range(N):
        P = PmW_array[i]
        resFreq, eqPos, err = postPowerValues(sphereMass, t, spacing, QofS, P, S)
        resonantFrequency[i] = resFreq
        equilibriumPosition[i] = eqPos
        error[i] = err
    return resonantFrequency, equilibriumPosition, error

def getTotalForceCurveFromParameters(diameterInMicron, rmax_mm, PmW):
    """Get the Force graph"""

    """Experiment parameters go here (make the numbers be floats)"""
    f = 25.  # mm; distance between the lens and the focus
    Smax = 10.  # mm; highest the sphere will levitate as the distance from the focus
    spacing = 1. / 1000.  # mm; this is how far apart our data points will be for r, S, and K

    """Calculations"""
    sphereMass, S, QofS = get_z_QofS(diameterInMicron, f, rmax_mm, Smax, spacing)
    totalForce = getTotalForceCurveFromQofS(PmW, sphereMass, QofS)
    return S, totalForce

def getPowerFromSphereSize(diameterInMicron, equilibriumPositionInMicron = 100., NA = 0.04):
    """Solves for the power needed to keep the sphere at some height"""
    """Variables"""
    f = 25.  # mm; distance between the lens and the focus
    rmax = 2.*f*NA  # mm; beam intensity is 1/e^2 at r = rmax
    spacing = 1. / 1000.  # mm; this is how far apart our data points will be for r, S, and K

    """Calculated parameters"""
    sphereRadius = diameterInMicron / 2000. # mm
    r_pts = np.arange(spacing, rmax * np.sqrt(np.log(10.)), spacing) # mm
    r = r_pts.reshape((len(r_pts), 1)) # mm
    S = equilibriumPositionInMicron/1000
    sphereVolume = 4. * np.pi * (sphereRadius ** 3.) / 3.  # mm cubed
    downwardForce = massDensity * sphereVolume / 1000. * g  # Newtons

    phi = np.arctan(r / f)
    sinTheta = np.sin(phi) * S / sphereRadius
    isPhysical = np.abs(sinTheta) <= 1.
    theta = np.arcsin(sinTheta * isPhysical)
    sinThetaR = sinTheta / n
    thetaR = np.arcsin(sinThetaR * isPhysical)

    # reflection and transmission coefficients
    a = np.cos(theta)
    b = np.sqrt(1. - (np.sin(theta) / n) ** 2.)
    R = ((((n * a - b) / (n * a + b)) ** 2.) + (((a - n * b) / (a + n * b)) ** 2.)) / 2.
    T = 1.0 - R

    # unitless force in terms of r and S
    Td = (T ** 2.) / (1. + R ** 2. + 2. * R * np.cos(2. * thetaR))
    dQs_init = 1. + R * np.cos(2. * theta) - Td * ((np.cos(2. * theta - 2. * thetaR) + R * np.cos(2. * theta)))

    # set dQs to 0 everywhere that the sphere isn't.
    dQs = dQs_init * isPhysical * np.cos(phi)

    # integrate to get total force; jacobian is rdr; normalize with a gaussian
    thingToIntegrate = r * dQs * np.exp(-np.power(r, 2.) / (2. * np.power(rmax / 2., 2.)))
    QofS = 2 * np.pi * np.trapz(thingToIntegrate, x=r, axis=0)

    # now I have a QofS and a downwardForce, so downwardForce/QofS gives me constant
    # constant = n * P / c  in Newtons
    power = (c/n)*(downwardForce/QofS)
    return power[0]