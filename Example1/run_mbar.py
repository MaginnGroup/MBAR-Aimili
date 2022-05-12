#-------------------------------------------#
# Author: Cassiano Aimoli (aimoli@gmail.com)#
#-------------------------------------------#
# This script uses pyMBAR package to to calculate MBAR estimation of thermodynamic properties via ensemble fluctuations in NPT or NVT ensemble:
# M.R. Shirts, J.D. Chodera, J. Chem. Phys., 129 (2008) 124105.
#
# The standard estimation is also calculated and the uncertainties are obtained using error propagation equations using the Uncertainties module.
#
# The ideal contribution to the heat capacity is calculated using experimental correlations:
# CH4 - U. Setzmann, W. Wagner, J. Phys. Chem. Ref. Data, 20 (1991) 1061-1150
# CO2 - R. Span, W. Wagner, J. Phys. Chem. Ref. Data, 25 (1996) 1509-1596.
# Ar - C. Tegeler, R. Span, W. Wagner, J. Phys. Chem. Ref. Data, 28 (1999) 779-850
#=============================================================================================
# IMPORTS
#=============================================================================================
# Requires the Uncertainties package available at https://pypi.python.org/pypi/uncertainties/
# Requires the pyMBAR package available at https://simtk.org/home/pymbar
from uncertainties import unumpy
import numpy
from numpy import *
import pymbar
import timeseries
import commands
import os.path
import subprocess
import shutil
#=============================================================================================
# PARAMETERS (CHANGE ACCORDINGLY)
#=============================================================================================
# Define ensemble ('NPT' or 'NVT')
ensemble = 'NVT' 

# Turn on/off MBAR calculation
run_mbar = 'on'

# Subsampling ('off' for uncorrelated original data and 'on' to use subsampling)
subsampling = 'on'  

# Original simulation system size and composition
N_CH4 = 256 # Number of CH4 molecules on the simulations
N_CO2 = 0 # Number of CO2 molecules on the simulations
N_Ar = 0 # Number of Ar molecules on the simulations

#=============================================================================================
# PATHS (CHANGE ACCORDINGLY)
#=============================================================================================
# Define paths
directory = ""
processed_data = directory + 'processed_data/'
results = directory + 'results/'
plots = directory + 'plots/'
exptl = directory + 'exptl_data/'

# Prepare folders
if os.path.exists(plots):
    shutil.rmtree(plots)
    os.mkdir(plots)
else:
    os.mkdir(plots)

if not os.path.exists(results):
    os.mkdir(results)

if not os.path.exists(exptl):
    os.mkdir(exptl)
#=============================================================================================
# CONSTANTS
#=============================================================================================
# Convertion factors
kcal2J = 4184.0 # Convert kcal2J
A32m3  = 1E-30 # Convert A3 to m3
A2m = 1E-10 # Conver A to m
atm2MPa = 0.101325 #Convert atm2MPa
atm2Pa = 101325 #Convert atm2Pa
fs2s = 1E-15 #Convert fs to s
Cmass  = 0.0120107 # Carbon mass (kg/mol)
Omass = 0.0159994 # Oxigen mass (kg/mol)
Hmass = 0.00100794 # Hidrogen mass (kg/mol)
Armass = 0.039948 # Hidrogen mass (kg/mol)
Avog = 6.02214129E+23 # Avogadro number
kB_JK = 1.3806488E-23 # Boltzmann constant in J/K
kB_kcalmolK = kB_JK * Avog / kcal2J # Boltzmann constant in kcal/mol/K

# Ideal CH4 Cp correlation parameters (Setzmann)
n = [4.0016,0.008449,4.6942,3.4865,1.6572,1.4115]
theta = [0.0,648,1957,3895,5705,15080]
# Ideal CO2 Cp correlation parameters (Span)
Tc = 304.1282
a0 = [0.0,8.37304456,-3.70454304,2.50000000,1.99427042,0.62105248,0.41195293,1.04028922,0.08327678]
theta0 = [0.0,0.0,0.0,0.0,3.15163,6.11190,6.77708,11.32384,27.08792]
# Ideal Ar Cp correlation parameters (Tegeler)
cp0 = 0.5203333
#=============================================================================================
# FUNCTIONS
#=============================================================================================
def Cp_id_func(n,theta,T,a0,theta0,Tc,cp0):
# Ideal CH4 Cp correlation function (Setzmann)
    Cp_id_CH4 = (N_CH4*kB_JK*(n[0]+n[1]*(theta[1]/T[k])**2*exp(theta[1]/T[k])/(exp(theta[1]/T[k])-1)**2+n[2]*(theta[2]/T[k])**2*exp(theta[2]/T[k])/(exp(theta[2]/T[k])-1)**2+n[3]*(theta[3]/T[k])**2*exp(theta[3]/T[k])/(exp(theta[3]/T[k])-1)**2+n[4]*(theta[4]/T[k])**2*exp(theta[4]/T[k])/(exp(theta[4]/T[k])-1)**2+n[5]*(theta[5]/T[k])**2*exp(theta[5]/T[k])/(exp(theta[5]/T[k])-1)**2))

# Ideal CO2 Cp correlation function (Span)
    Cp_id_CO2 = (N_CO2*kB_JK*(1+a0[3]+a0[4]*(theta0[4]*Tc/T[k])**2*(exp(theta0[4]*Tc/T[k]))/(exp(theta0[4]*Tc/T[k])-1)**2+a0[5]*(theta0[5]*Tc/T[k])**2*(exp(theta0[5]*Tc/T[k]))/(exp(theta0[5]*Tc/T[k])-1)**2+a0[6]*(theta0[6]*Tc/T[k])**2*(exp(theta0[6]*Tc/T[k]))/(exp(theta0[6]*Tc/T[k])-1)**2+a0[7]*(theta0[7]*Tc/T[k])**2*(exp(theta0[7]*Tc/T[k]))/(exp(theta0[7]*Tc/T[k])-1)**2+a0[8]*(theta0[8]*Tc/T[k])**2*(exp(theta0[8]*Tc/T[k]))/(exp(theta0[8]*Tc/T[k])-1)**2))

# Ideal Ar Cp correlation function (Tegeler)
    Cp_id_Ar = (N_Ar*cp0*1000*Armass/Avog)

    return Cp_id_CH4 + Cp_id_CO2 + Cp_id_Ar

# Total number of molecules
N_total = N_CH4 + N_CO2 + N_Ar 

# rho (kg/m3)
def rho_func(V):
    return (((Cmass+2*Omass)*N_CO2+(Cmass+4*Hmass)*N_CH4+(Armass)*N_Ar)/Avog/(V*1E-30))

# aP (1/K)
def aP_func(VH, V, H, T):
    return ((VH-V*H)/(V*kB_kcalmolK*T*T))
        
# kT (1/Pa)
def kT_func(V2,V,T):
    return ((V2-V*V)/(V*kB_JK*T)*A32m3)
        
# Cp (J/K)
def Cp_func(UH,U,H,T,VH,V,P,Cp_id):
    return (((UH-U*H)*(kcal2J/Avog)**2/(kB_JK*T*T))+((VH-V*H)*(kcal2J/Avog*A32m3)*(P*1E6)/(kB_JK*T*T))-((N_total)*kB_JK)) + Cp_id
        
# Cv_NPT (J/K)
def Cv_NPT_func(Cp,T,V,aP,kT):
    return (Cp-T*(V*A32m3)*aP**2/kT)
        
# uJT (K/Pa)
def uJT_func(V,Cp,T,aP):
    return ((V*A32m3)/Cp*(T*aP-1))
        
# SS (m/s)
def SS_func(Cp,Cv,kT,rho):   
    return (Cp/Cv/kT/rho)

# Pressure_NVT (MPa)
def P_NVT_func(P,T,V):
    return (P+(N_total)*kB_JK*T/(V*1E-30)*1E-6)

# Cv_NVT (J/K)
def Cv_NVT_func(U2,U,T,Cv_id):
    return (U2-U*U)*(kcal2J/Avog)**2/(kB_JK*T*T)+Cv_id

# Fit function for pressure and heat flux correlation functions
def fit_func(x, a, b):
    return a*(1-exp(-x/b))
#=============================================================================================
# MAIN
#=============================================================================================
print "Reading input files..."
# Determine maximum number of snapshots in all simulations
filename = os.path.join(processed_data + 'uconf.dat')
T_max = int(commands.getoutput('wc -l %s' % filename).split()[0])

# READ DATA
# Read Temperature
filename = os.path.join(processed_data + 'temperature.dat')
print "Reading %s..." % filename
infile = open(filename, 'r')
elements = infile.readline().split()
K = len(elements)
temperature = zeros([K], float64)
for k in range(K):
    temperature[k] = float(elements[k])
infile.close()
        
# Read Volume
if ensemble == 'NPT':
    T_k = zeros([K], int64)
    volume_original = zeros([K,T_max], float64)
    filename = os.path.join(processed_data + 'volume.dat')
    print "Reading %s..." % filename
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
    for line in lines:
        elements = line.split()
        for k in range(K):
            t = T_k[k]
            volume_original[k,t] = float(elements[k])
            T_k[k] += 1

if ensemble == 'NVT':
    filename = os.path.join(processed_data + 'volume.dat')
    print "Reading %s..." % filename
    infile = open(filename, 'r')
    elements = infile.readline().split()
    volume = zeros([K], float64)
    for k in range(K):
        volume[k] = float(elements[k])
    infile.close()

# Read Pressure
if ensemble == 'NPT':
    filename = os.path.join(processed_data + 'pressure.dat')
    print "Reading %s..." % filename
    infile = open(filename, 'r')
    elements = infile.readline().split()
    pressure = zeros([K], float64)
    for k in range(K):
        pressure[k] = float(elements[k])
    infile.close()

if ensemble == 'NVT':
    T_k = zeros([K], int64)
    pressure_original = zeros([K,T_max], float64)
    filename = os.path.join(processed_data + 'pressure.dat')
    print "Reading %s..." % filename
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
    for line in lines:
        elements = line.split()
        for k in range(K):
            t = T_k[k]
            pressure_original[k,t] = float(elements[k])
            T_k[k] += 1
    # Calculate residual pressure (subtract ideal contribution)
    for ind in range(K):
        for k in range(T_max):
            pressure_original[ind,k] = pressure_original[ind,k]*atm2MPa-(N_total)*kB_JK*temperature[ind]/(volume[ind]*1E-30)*1E-6
        
# Read Uconf
T_k = zeros([K], int64)
uconf_original = zeros([K,T_max], float64)
filename = os.path.join(processed_data + 'uconf.dat')
print "Reading %s..." % filename
infile = open(filename, 'r')
lines = infile.readlines()
infile.close()        
for line in lines:
    elements = line.split()
    for k in range(K):
        t = T_k[k]
        uconf_original[k,t] = float(elements[k])
        T_k[k] += 1

# Subsample trajectories based on the smallest number of independent samples from Uconf or Volume (NPT) or Pressure (NVT)
uconf = zeros([K, T_max], float64)
N_k = zeros([K], int64)

if ensemble == 'NPT':
    volume = zeros([K, T_max], float64)

if ensemble == 'NVT':
    pressure = zeros([K, T_max], float64)
       
if subsampling == 'on':
    print "Subsampling data..."
    uconf = zeros([K, T_max], float64)
    N_ksam = zeros([K], int64)
    indices2 = zeros([T_max], int64)
    for k in range(1,T_max):
        indices2[k] = k
    for k in range(K):
        # Compute correlation times
        if not uconf_original[k,0] == 0:
            ind1 = timeseries.subsampleCorrelatedData(uconf_original[k,0:T_k[k]])
            if ensemble == 'NPT':
                ind2 = timeseries.subsampleCorrelatedData(volume_original[k,0:T_k[k]])
            if ensemble == 'NVT':
                ind2 = timeseries.subsampleCorrelatedData(pressure_original[k,0:T_k[k]])
            if len(ind1) == min(len(ind1),len(ind2)):
                indices = ind1
            if len(ind2) == min(len(ind1),len(ind2)):
                indices = ind2
            # Store subsampled positions
            if len(indices) >= 1000:
                N_ksam[k] = len(indices)
                if ensemble == 'NPT':
                    volume[k,0:N_ksam[k]] = volume_original[k,indices]
                if ensemble == 'NVT':
                    pressure[k,0:N_ksam[k]] = pressure_original[k,indices]
                uconf[k,0:N_ksam[k]] = uconf_original[k,indices]
            else:
                N_ksam[k] = len(indices2)
                if ensemble == 'NPT':
                    volume[k,0:N_ksam[k]] = volume_original[k,indices2]
                if ensemble == 'NVT':
                    pressure[k,0:N_ksam[k]] = pressure_original[k,indices2]
                uconf[k,0:N_ksam[k]] = uconf_original[k,indices2]
            print('\n')
        else:
            N_ksam[k] = len(indices2)
            if ensemble == 'NPT':
                volume[k,0:N_ksam[k]] = volume_original[k,indices2]
            if ensemble == 'NVT':
                pressure[k,0:N_ksam[k]] = pressure_original[k,indices2]
            uconf[k,0:N_ksam[k]] = uconf_original[k,indices2]       
    N_k = N_ksam
else:
    print "Using all data provided without subsampling..."
    if ensemble == 'NPT':
        volume = volume_original
    if ensemble == 'NVT':
        pressure = pressure_original
    uconf = uconf_original
    for k in range(K):
        N_k[k] = T_max
 
for k in range(K):
    if uconf[k,0] == 0:
        N_k[k] = 0

# Calculate additional variables
if ensemble == 'NPT':
    hconf,volume2,vol_hconf,uconf_hconf = (numpy.zeros([K,T_max], numpy.float64) for m in range(4))
    for k in range(K):
        hconf[k,0:N_k[k]] = uconf[k,0:N_k[k]] + pressure[k]*1E6*volume[k,0:N_k[k]]*A32m3/kcal2J*Avog 
        volume2[k,0:N_k[k]] = volume[k,0:N_k[k]]*volume[k,0:N_k[k]]
        vol_hconf[k,0:N_k[k]] = volume[k,0:N_k[k]]*hconf[k,0:N_k[k]]
        uconf_hconf[k,0:N_k[k]] = uconf[k,0:N_k[k]]*hconf[k,0:N_k[k]]

if ensemble == 'NVT':
    uconf2 = numpy.zeros([K,T_max], numpy.float64)
    for k in range(K):
        uconf2[k,0:N_k[k]] = uconf[k,0:N_k[k]]*uconf[k,0:N_k[k]]

if run_mbar == 'on':
    # RUNNING MBAR
    # Calculate reduced potentials
    beta_k = (kB_kcalmolK * temperature)**(-1)
    u_kln = zeros([K,K,max(N_k)], float64)

    if ensemble == 'NPT':      
        for k in range(K):
            for l in range(K):
                u_kln[k,l,0:N_k[k]] = beta_k[l] * (uconf[k,0:N_k[k]] + pressure[l]*1E6*volume[k,0:N_k[k]]*A32m3/kcal2J*Avog)

    if ensemble == 'NVT':
        for k in range(K):
            for l in range(K):
                u_kln[k,l,0:N_k[k]] = beta_k[l] * uconf[k,0:N_k[k]]

    # Read initial guesss for f_k (if any)        
    if os.path.exists(results + 'f_k.dat'):
        infile = open(results + 'f_k.dat', 'r')
        lines = infile.readlines()
        infile.close()
        elements = list()
        for line in lines:
            elements += line.split()
        if len(elements) == len(temperature):
            f_k = numpy.zeros([K], numpy.float64)
            for k in range(K):
                f_k[k] = float(elements[k])

            # Initialize MBAR
            print "Running MBAR..."
            print " "
            print "Reading free energies from f_k.dat"
            mbar = pymbar.MBAR(u_kln, N_k, verbose = True, method = 'adaptive', relative_tolerance = 1.0e-10, initial_f_k = f_k)
        else:
            print "Running MBAR..."
            mbar = pymbar.MBAR(u_kln, N_k, verbose = True, method = 'adaptive', relative_tolerance = 1.0e-10)
    else:
        print "Running MBAR..."
        mbar = pymbar.MBAR(u_kln, N_k, verbose = True, method = 'adaptive', relative_tolerance = 1.0e-10)

    # Using MBAR to estimate observables
    print "Calculating observables"
    print "Uconf..."
    (uconf_MBAR, duconf_MBAR) = mbar.computeExpectations(uconf)

    if ensemble == 'NPT':
        hconfm = zeros([K,K,T_max], float64)
        vol_hconfm = zeros([K,K,T_max], float64)
        uconf_hconfm = zeros([K,K,T_max], float64)
        
        for k in range(K):
            for l in range(K):
                hconfm[k,l,0:N_k[k]] = uconf[k,0:N_k[k]] + pressure[l]*1E6*volume[k,0:N_k[k]]*A32m3/kcal2J*Avog
                vol_hconfm[k,l,0:N_k[k]] = volume[k,0:N_k[k]] * hconfm[k,l,0:N_k[k]]
                uconf_hconfm[k,l,0:N_k[k]] = uconf[k,0:N_k[k]] * hconfm[k,l,0:N_k[k]]
        
        print "Hconf..."
        (hconf_MBAR, dhconf_MBAR) = mbar.computeExpectations(hconfm)
        print "Volume..."
        (volume_MBAR, dvolume_MBAR) = mbar.computeExpectations(volume)
        print "Volume^2..."
        (volume2_MBAR, dvolume2_MBAR) = mbar.computeExpectations(volume2)
        print "Volume x Hconf..."
        (vol_hconf_MBAR, dvol_hconf_MBAR) = mbar.computeExpectations(vol_hconfm)
        print "Uconf x Hconf..."
        (uconf_hconf_MBAR, duconf_hconf_MBAR) = mbar.computeExpectations(uconf_hconfm)

    if ensemble == 'NVT':
        print "Pressure..."
        (pressure_conf_MBAR, dpressure_conf_MBAR) = mbar.computeExpectations(pressure)
        print "Uconf^2..."
        (uconf2_MBAR, duconf2_MBAR) = mbar.computeExpectations(uconf2)

# COMPUTING OBSERVABLES
# Preparing arrays for standard observables calculation (removing columns of zeros, if any)
i = 0
for k in range(K):
    if not uconf[k,0] == 0:
        i += 1

N_k_std,temperature_std = (numpy.zeros([i], numpy.float64) for m in range(2))
uconf_std = numpy.zeros([i, T_max], numpy.float64)
        
if ensemble == 'NPT':
    hconf_std,volume_std,volume2_std,vol_hconf_std,uconf_hconf_std = (numpy.zeros([i, T_max], numpy.float64) for m in range(5))
    pressure_std,hconf_STD,dhconf_STD,volume_STD,dvolume_STD,volume2_STD,dvolume2_STD,vol_hconf_STD,dvol_hconf_STD,uconf_STD,duconf_STD,uconf_hconf_STD,duconf_hconf_STD = (numpy.zeros([i], numpy.float64) for m in range(13))

if ensemble == 'NVT':
    pressure_std,uconf2_std = (numpy.zeros([i, T_max], numpy.float64)for m in range(2))
    volume_std,uconf_STD,duconf_STD,pressure_conf_STD,dpressure_conf_STD,uconf2_STD,duconf2_STD = (numpy.zeros([i], numpy.float64) for m in range(7))
        
# Removing columns of zeros
if ensemble == 'NPT':
    I = 0
    for k in range(K):
        if not hconf[k,0] == 0:
            hconf_std[I] = hconf[k]
            N_k_std[I] = N_k[k]
            temperature_std[I] = temperature[k]
            pressure_std[I] = pressure[k]
            volume_std[I] = volume[k]
            volume2_std[I] = volume2[k]
            vol_hconf_std[I] = vol_hconf[k]
            uconf_std[I] = uconf[k]
            uconf_hconf_std[I] = uconf_hconf[k]
            I += 1

if ensemble == 'NVT':
    I = 0
    for k in range(K):
        if not uconf[k,0] == 0:
            uconf_std[I] = uconf[k]
            N_k_std[I] = N_k[k]
            temperature_std[I] = temperature[k]
            volume_std[I] = volume[k]
            pressure_std[I] = pressure[k]
            uconf2_std[I] = uconf2[k]
            I += 1
        
# Calculating standard estimates of observables and deviations
if ensemble == 'NPT':
    for k in range(I):
        hconf_STD[k] = numpy.average(hconf_std[k,0:N_k_std[k]])
        dhconf_STD[k]  = numpy.sqrt(numpy.var(hconf_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        volume_STD[k] = numpy.average(volume_std[k,0:N_k_std[k]])
        dvolume_STD[k]  = numpy.sqrt(numpy.var(volume_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        volume2_STD[k] = numpy.average(volume2_std[k,0:N_k_std[k]])
        dvolume2_STD[k]  = numpy.sqrt(numpy.var(volume2_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        vol_hconf_STD[k] = numpy.average(vol_hconf_std[k,0:N_k_std[k]])
        dvol_hconf_STD[k]  = numpy.sqrt(numpy.var(vol_hconf_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        uconf_STD[k] = numpy.average(uconf_std[k,0:N_k_std[k]])
        duconf_STD[k]  = numpy.sqrt(numpy.var(uconf_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        uconf_hconf_STD[k] = numpy.average(uconf_hconf_std[k,0:N_k_std[k]])
        duconf_hconf_STD[k]  = numpy.sqrt(numpy.var(uconf_hconf_std[k,0:N_k_std[k]])/(N_k_std[k]-1))

if ensemble == 'NVT':
    for k in range(I):
        uconf_STD[k] = numpy.average(uconf_std[k,0:N_k_std[k]])
        duconf_STD[k]  = numpy.sqrt(numpy.var(uconf_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        pressure_conf_STD[k] = numpy.average(pressure_std[k,0:N_k_std[k]])
        dpressure_conf_STD[k]  = numpy.sqrt(numpy.var(pressure_std[k,0:N_k_std[k]])/(N_k_std[k]-1))
        uconf2_STD[k] = numpy.average(uconf2_std[k,0:N_k_std[k]])
        duconf2_STD[k]  = numpy.sqrt(numpy.var(uconf2_std[k,0:N_k_std[k]])/(N_k_std[k]-1))

# CALCULATING PROPERTIES
# Preparing arrays
temperature_dev,dtemperature = (numpy.zeros([K],numpy.float64) for m in range(2))
temperature_std_dev,dtemperature_std,uconf_STDdev,rho_STD,Cv_STDdev = (numpy.zeros([I],numpy.float64) for m in range(5))

if run_mbar == 'on':
    uconf_MBARdev,rho_MBAR,Cv_MBARdev = (numpy.zeros([K],numpy.float64) for m in range(3))

if ensemble == 'NPT':
    pressure_dev,dpressure = (numpy.zeros([K],numpy.float64) for m in range(2))
    pressure_std_dev,dpressure_std,uconf_hconf_STDdev,duconf_hconf_STDdev,hconf_STDdev,volume_STDdev,volume2_STDdev,vol_hconf2_STDdev,aP_STDdev,kT_STDdev,Cp_STDdev,Cp_id_STD,uJT_STDdev,SS_STDdev = (numpy.zeros([I],numpy.float64) for m in range(14))

    if run_mbar == 'on':
        hconf_MBARdev,volume_MBARdev,volume2_MBARdev,vol_hconf2_MBARdev,aP_MBARdev,kT_MBARdev,Cp_MBARdev,Cp_id_MBAR,uJT_MBARdev,SS_MBARdev = (numpy.zeros([K],numpy.float64) for m in range(10))

if ensemble == 'NVT':
    volume_dev,dvolume = (numpy.zeros([K],numpy.float64) for m in range(2))
    volume_std_dev,dvolume_std,uconf2_STDdev,pressure_conf_STDdev,Cv_id_STD,Cv_id_STD2,pressure_STDdev = (numpy.zeros([I],numpy.float64) for m in range(7))

    if run_mbar == 'on':
        uconf2_MBARdev,pressure_conf_MBARdev,Cv_id_MBAR,pressure_MBARdev = (numpy.zeros([K],numpy.float64) for m in range(4))
        
# Concatenate averages and deviations into the same array
temperature_dev = unumpy.uarray(temperature,dtemperature)
temperature_std_dev = unumpy.uarray(temperature_std,dtemperature_std)

uconf_STDdev = unumpy.uarray(uconf_STD,duconf_STD)

if run_mbar == 'on': 
    uconf_MBARdev = unumpy.uarray(uconf_MBAR,duconf_MBAR)

if ensemble == 'NPT':
    pressure_dev = unumpy.uarray(pressure,dpressure)
    pressure_std_dev = unumpy.uarray(pressure_std,dpressure_std)
        
    hconf_STDdev = unumpy.uarray(hconf_STD,dhconf_STD)
    hconf_STDdev = unumpy.uarray(hconf_STD,dhconf_STD)
    volume_STDdev = unumpy.uarray(volume_STD,dvolume_STD)
    volume2_STDdev = unumpy.uarray(volume2_STD,dvolume2_STD)
    vol_hconf_STDdev = unumpy.uarray(vol_hconf_STD,dvol_hconf_STD)
    uconf_hconf_STDdev = unumpy.uarray(uconf_hconf_STD,duconf_hconf_STD)

    if run_mbar == 'on':
        hconf_MBARdev = unumpy.uarray(hconf_MBAR,dhconf_MBAR)
        hconf_MBARdev = unumpy.uarray(hconf_MBAR,dhconf_MBAR)
        volume_MBARdev = unumpy.uarray(volume_MBAR,dvolume_MBAR)
        volume2_MBARdev = unumpy.uarray(volume2_MBAR,dvolume2_MBAR)
        vol_hconf_MBARdev = unumpy.uarray(vol_hconf_MBAR,dvol_hconf_MBAR)
        uconf_hconf_MBARdev = unumpy.uarray(uconf_hconf_MBAR,duconf_hconf_MBAR)

    # rho (kg/m3)
    rho_STDdev = rho_func(volume_STDdev)
    # aP (1/K)
    aP_STDdev = aP_func(vol_hconf_STDdev,volume_STDdev,hconf_STDdev,temperature_std_dev)
    # kT (1/Pa)
    kT_STDdev = kT_func(volume2_STDdev,volume_STDdev,temperature_std_dev)
    # Cp_id (J/K)
    for k in range(I):
        Cp_id_STD[k] = Cp_id_func(n,theta,temperature,a0,theta0,Tc,cp0)
    # Cp (J/K)
    Cp_STDdev = Cp_func(uconf_hconf_STDdev,uconf_STDdev,hconf_STDdev,temperature_std_dev,vol_hconf_STDdev,volume_STDdev,pressure_std_dev,Cp_id_STD)
    # Cv (1/Pa)
    Cv_STDdev = Cv_NPT_func(Cp_STDdev,temperature_std_dev,volume_STDdev,aP_STDdev,kT_STDdev)
    # uJT (K/Pa)
    uJT_STDdev = uJT_func(volume_STDdev,Cp_STDdev,temperature_std_dev,aP_STDdev)
    # SS (m/s) - Sets the value to zero in case of unphysical behaviour that might lead to sqrt of negative numbers
    SS_STDdev = SS_func(Cp_STDdev,Cv_STDdev,kT_STDdev,rho_STDdev)  
    for k in range(I):
        if unumpy.nominal_values(SS_STDdev[k]) >= 0.0:
            SS_STDdev[k] = SS_STDdev[k]**.5
        else:
            SS_STDdev[k] = 0.0
              
    # Convert units
    kT_STDdev = kT_STDdev*1E6
    Cp_id_STD = Cp_id_STD*Avog/(N_total)
    Cp_STDdev = Cp_STDdev*Avog/(N_total)
    Cv_STDdev = Cv_STDdev*Avog/(N_total)
    uJT_STDdev = uJT_STDdev*1E6

    if run_mbar == 'on':
        # rho (kg/m3)
        rho_MBARdev = rho_func(volume_MBARdev)
        # aP (1/K)
        aP_MBARdev = aP_func(vol_hconf_MBARdev,volume_MBARdev,hconf_MBARdev,temperature_dev)
        # kT (1/Pa)
        kT_MBARdev = kT_func(volume2_MBARdev,volume_MBARdev,temperature_dev)
        # Cp_id (J/K)
        for k in range(K):
            Cp_id_MBAR[k] = Cp_id_func(n,theta,temperature,a0,theta0,Tc,cp0)
        # Cp (J/K)
        Cp_MBARdev = Cp_func(uconf_hconf_MBARdev,uconf_MBARdev,hconf_MBARdev,temperature_dev,vol_hconf_MBARdev,volume_MBARdev,pressure_dev,Cp_id_MBAR)
        # Cv (1/Pa)
        Cv_MBARdev = Cv_NPT_func(Cp_MBARdev,temperature_dev,volume_MBARdev,aP_MBARdev,kT_MBARdev)
        # uJT (K/Pa)
        uJT_MBARdev = uJT_func(volume_MBARdev,Cp_MBARdev,temperature_dev,aP_MBARdev)
        # SS (m/s) - Sets the value to zero in case of unphysical behaviour that might lead to sqrt of negative numbers
        SS_MBARdev = SS_func(Cp_MBARdev,Cv_MBARdev,kT_MBARdev,rho_MBARdev)  
        for k in range(K):
            if unumpy.nominal_values(SS_MBARdev[k]) >= 0.0:
                SS_MBARdev[k] = SS_MBARdev[k]**.5
            else:
                SS_MBARdev[k] = 0.0
        
        # Convert units
        kT_MBARdev = kT_MBARdev*1E6
        Cp_id_MBAR = Cp_id_MBAR*Avog/(N_total)
        Cp_MBARdev = Cp_MBARdev*Avog/(N_total)
        Cv_MBARdev = Cv_MBARdev*Avog/(N_total)
        uJT_MBARdev = uJT_MBARdev*1E6

if ensemble == 'NVT':
    volume_dev = unumpy.uarray(volume,dvolume)
    volume_std_dev = unumpy.uarray(volume_std,dvolume_std)
    uconf2_STDdev = unumpy.uarray(uconf2_STD,duconf2_STD)
    pressure_conf_STDdev = unumpy.uarray(pressure_conf_STD,dpressure_conf_STD)

    # Pressure (MPa)
    pressure_STDdev = P_NVT_func(pressure_conf_STDdev,temperature_std_dev,volume_std_dev)
    # rho (kg/m3)
    rho_STD = rho_func(volume_std_dev)
    # Cv_id - Experimental correlation Setzmann for CH4 (J/K) - Cv = Cp - R
    for k in range(I):
        Cv_id_STD[k] = Cp_id_func(n,theta,temperature,a0,theta0,Tc,cp0)-(N_total)*kB_JK
    # Cv (J/K)
    Cv_STDdev = Cv_NVT_func(uconf2_STDdev,uconf_STDdev,temperature_std_dev,Cv_id_STD)
    # Convert units
    Cv_id_STD = Cv_id_STD*Avog/(N_total)
    Cv_STDdev = Cv_STDdev*Avog/(N_total)

    if run_mbar == 'on':
        uconf2_MBARdev = unumpy.uarray(uconf2_MBAR,duconf2_MBAR)
        pressure_conf_MBARdev = unumpy.uarray(pressure_conf_MBAR,dpressure_conf_MBAR)

        # Pressure (MPa)
        pressure_MBARdev = P_NVT_func(pressure_conf_MBARdev,temperature_dev,volume_dev)
        # rho (kg/m3)
        rho_MBAR = rho_func(volume)
        # Cv_id - Experimental correlation Setzmann for CH4 (J/K) - Cv = Cp - R
        for k in range(K):
            Cv_id_MBAR[k] = Cp_id_func(n,theta,temperature,a0,theta0,Tc,cp0)-(N_total)*kB_JK
        # Cv (J/K)
        Cv_MBARdev = Cv_NVT_func(uconf2_MBARdev,uconf_MBARdev,temperature_dev,Cv_id_MBAR)
        # Convert units
        Cv_id_MBAR = Cv_id_MBAR*Avog/(N_total)
        Cv_MBARdev = Cv_MBARdev*Avog/(N_total)

# OUTPUT RESULTS
if ensemble == 'NPT':
    # Write Thermodynamic Properties file (NPT - Standard)
    filename = os.path.join(results + 'STD_results.dat')
    print "Writing final results to '%s'..." % filename
    outfile = open(filename, 'w')
    outfile.write('Temperature(K) Pressure(MPa) Hconf(kcal/mol) dHconf Volume(A3) dV V^2(A6) dV^2 V*Hconf(A3*kcal/mol) dV*Hconf Uconf(kcal/mol) dUconf Uconf*Hconf(kcal/mol) dUconf*Hconf rho(kg/m3) drho aP(1/K) daP kT(1/MPa) dkT Cp_id(J/molK) Cp(J/molK) dCp Cv(J/molK) dCv uJT(K/MPa) duJT SS(m/s) dSS')
    outfile.write('\n')
    for k in range(I):
        outfile.write('%.10e' % unumpy.nominal_values(temperature_std_dev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(pressure_std_dev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(hconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(hconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(volume_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(volume_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(volume2_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(volume2_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(vol_hconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(vol_hconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(uconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(uconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(uconf_hconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(uconf_hconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(rho_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(rho_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(aP_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(aP_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(kT_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(kT_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(Cp_id_STD[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(Cp_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(Cp_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(Cv_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(Cv_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(uJT_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(uJT_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(SS_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(SS_STDdev[k]))
        outfile.write('\n')
    outfile.close()

    if run_mbar == 'on':
        # Write Thermodynamic Properties file (NPT - MBAR)     
        filename = os.path.join(results + 'MBAR_results.dat')
        print "Writing final results to '%s'..." % filename
        outfile = open(filename, 'w')
        outfile.write('Temperature(K) Pressure(MPa) Hconf(kcal/mol) dHconf Volume(A3) dV V^2(A6) dV^2 V*Hconf(A3*kcal/mol) dV*Hconf Uconf(kcal/mol) dUconf Uconf*Hconf(kcal/mol) dUconf*Hconf rho(kg/m3) drho aP(1/K) daP kT(1/MPa) dkT Cp_id(J/molK) Cp(J/molK) dCp Cv(J/molK) dCv uJT(K/MPa) duJT SS(m/s) dSS')
        outfile.write('\n')
        for k in range(K):
            outfile.write('%.10e' % unumpy.nominal_values(temperature_dev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(pressure_dev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(hconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(hconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(volume_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(volume_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(volume2_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(volume2_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(vol_hconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(vol_hconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(uconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(uconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(uconf_hconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(uconf_hconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(rho_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(rho_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(aP_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(aP_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(kT_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(kT_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(Cp_id_MBAR[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(Cp_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(Cp_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(Cv_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(Cv_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(uJT_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(uJT_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(SS_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(SS_MBARdev[k]))
            outfile.write('\n')
        outfile.close()

if ensemble == 'NVT':
    # Write Thermodynamic Properties file (NVT - Standard)
    filename = os.path.join(results + 'STD_results.dat')
    print "Writing final results to '%s'..." % filename
    outfile = open(filename, 'w')
    outfile.write('Temperature(K) Volume(A3) Pressure(MPa) dPressure Uconf(kcal/mol) dUconf Uconf2(kcal/mol)^2 dUconf2 rho(kg/m3) Cv_id(J/molK) Cv(J/molK) dCv')
    outfile.write('\n')
    for k in range(I):
        outfile.write('%.10e' % unumpy.nominal_values(temperature_std[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(volume_std[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(pressure_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(pressure_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(uconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(uconf_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(uconf2_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(uconf2_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(rho_STD[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(Cv_id_STD[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.nominal_values(Cv_STDdev[k]))
        outfile.write(' ')
        outfile.write('%.10e' % unumpy.std_devs(Cv_STDdev[k]))
        outfile.write('\n')
    outfile.close()

    if run_mbar == 'on': 
        # Write Thermodynamic Properties file (NVT - MBAR)
        filename = os.path.join(results + 'MBAR_results.dat')
        print "Writing final results to '%s'..." % filename
        outfile = open(filename, 'w')
        outfile.write('Temperature(K) Volume(A3) Pressure(MPa) dPressure Uconf(kcal/mol) dUconf Uconf2(kcal/mol)^2 dUconf2 rho(kg/m3) Cv_id(J/molK) Cv(J/molK) dCv')
        outfile.write('\n')
        for k in range(K):
            outfile.write('%.10e' % unumpy.nominal_values(temperature[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(volume[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(pressure_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(pressure_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(uconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(uconf_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(uconf2_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(uconf2_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(rho_MBAR[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(Cv_id_MBAR[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.nominal_values(Cv_MBARdev[k]))
            outfile.write(' ')
            outfile.write('%.10e' % unumpy.std_devs(Cv_MBARdev[k]))
            outfile.write('\n')
        outfile.close()

if run_mbar == 'on':
    # Write Free Energies file
    print "Writing free energies"
    f_k=mbar.f_k
    K = f_k.size
    outfile = open(results + 'f_k.dat', 'w')
    for k in range(K):
        outfile.write("%.10e " % f_k[k])
    outfile.write("\n")
    outfile.close()

# GNUPLOTS
# Prepare folder
if os.path.exists(plots):
    shutil.rmtree(plots)
    os.mkdir(plots)
else:
    os.mkdir(plots)

# Define file paths
thermo_std_data  = results + 'STD_results.dat'
thermo_mbar_data  = results + 'MBAR_results.dat'
nist_data = exptl + "NIST.dat"

# make gnuplot plots
gnuplot_in = os.path.join(plots + 'gnuplot.in')

if ensemble == 'NVT':
    # Pressure
    filename = os.path.join(plots, '01_pressure.eps')
    gnuplot_input = """
        set term postscript solid
        set output "%(filename)s"
        set title "Pressure"
        set xlabel "Temperature (K)"
        set ylabel "Pressure (MPa)"
        plot "%(thermo_std_data)s" u 1:3:4 with yerrorbars t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:3:4 with yerrorbars t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:2 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())
    
    # Isochoric Heat Capacity
    filename = os.path.join(plots, '02_Cv.eps')
    gnuplot_input = """
        set term postscript solid
        set output "%(filename)s"
        set title "Isochoric Heat Capacity"
        set xlabel "Temperature (K)"
        set ylabel "Cv (J/mol.K)"
        plot "%(thermo_std_data)s" u 1:11:12 with yerrorbars t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:11:12 with yerrorbars t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:12 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())
    output = commands.getoutput('rm %(gnuplot_in)s' % vars())

if ensemble == 'NPT':
    # Density
    filename = os.path.join(plots, '01_density.eps')
    gnuplot_input = """
        set term postscript solid
        set output "%(filename)s"
        set title "Density"
        set xlabel "Temperature (K)"
        set ylabel "Density (kg/m^3)"
        plot "%(thermo_std_data)s" u 1:15 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:15 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:3 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())

    # Coefficient of Thermal Expansion
    filename = os.path.join(plots, '02_aP.eps')
    gnuplot_input = """
        set term postscript color solid
        set output "%(filename)s"
        set title "Coefficient of Thermal Expansion"
        set xlabel "Temperature (K)"
        set ylabel "aP (1/K)"
        plot "%(thermo_std_data)s" u 1:17 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:17 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:(sqrt(($9-$8)*($9/$8/$3/$10/$10)/$1/($4*((%(Cmass)s+2*%(Omass)s)*%(N_CO2)s+(%(Cmass)s+4*%(Hmass)s)*%(N_CH4)s+(%(Armass)s)*%(N_Ar)s)/%(N_total)s))) t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())

    # Isothermal Compressibility
    filename = os.path.join(plots, '03_kT.eps')
    gnuplot_input = """
        set term postscript color solid
        set output "%(filename)s"
        set title "Isothermal Compressibility"
        set xlabel "Temperature (K)"
        set ylabel "aP (1/MPa)"
        plot "%(thermo_std_data)s" u 1:19 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:19 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:($9/$8/$3/$10/$10*1000000) t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())

    # Isobaric Heat Capacity
    filename = os.path.join(plots, '04_Cp.eps')
    gnuplot_input = """
        set term postscript color solid
        set output "%(filename)s"
        set title "Isobaric Heat Capacity"
        set xlabel "Temperature (K)"
        set ylabel "Cp (J/mol.K)"
        plot "%(thermo_std_data)s" u 1:22 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:22 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:9 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())

    # Isochoric Heat Capacity
    filename = os.path.join(plots, '05_Cv.eps')
    gnuplot_input = """
        set term postscript color solid
        set output "%(filename)s"
        set title "Isochoric Heat Capacity"
        set xlabel "Temperature (K)"
        set ylabel "Cv (J/mol.K)"
        plot "%(thermo_std_data)s" u 1:24 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:24 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:8 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())

    # Joule-Thomson Coefficient
    filename = os.path.join(plots, '06_uJT.eps')
    gnuplot_input = """
        set term postscript color solid
        set output "%(filename)s"
        set title "Joule-Thomson Coefficient"
        set xlabel "Temperature (K)"
        set ylabel "uJT (K/MPa)"
        plot "%(thermo_std_data)s" u 1:26 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:26 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:11 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())

    # Speed of Sound
    filename = os.path.join(plots, '07_SS.eps')
    gnuplot_input = """
        set term postscript color solid
        set output "%(filename)s"
        set title "Speed of Sound"
        set xlabel "Temperature (K)"
        set ylabel "SS (m/s)"
        plot "%(thermo_std_data)s" u 1:28 t "STD estimate" pt 7 lc rgb "red" ps 1, "%(thermo_mbar_data)s" u 1:28 t "MBAR optimal estimate" pt 6 lc rgb "black" ps 1, "%(nist_data)s" u 1:10 t "EXP" pt 13 lc rgb "blue" ps 0.5
        """ % vars()
    outfile = open(gnuplot_in, 'w')
    outfile.write(gnuplot_input)
    outfile.close()
    output = commands.getoutput('gnuplot < %(gnuplot_in)s' % vars())
    output = commands.getoutput('rm %(gnuplot_in)s' % vars())

print "DONE!"
