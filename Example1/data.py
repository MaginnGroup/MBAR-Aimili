#-------------------------------------------#
# Author: Cassiano Aimoli (aimoli@gmail.com)#
#-------------------------------------------#
# This script extracts data from LAMMPS log files and build the files to be used as inputs of the run_mbar.py script.
#=============================================================================================
# IMPORTS
#=============================================================================================
import numpy
from numpy import *
import commands
import os
import os.path
import shutil
#=============================================================================================
# PARAMETERS (CHANGE ACCORDINGLY)
#=============================================================================================
# Define ensemble ('NPT' or 'NVT')
ensemble = 'NVT'

# Define unsampled states ('yes' or 'no')
include_unsampled = 'no' # if 'yes', specify T (K) and P (MPa - only NPT ensemble) for new states
unsampled_T = [305.15,310.15,315.15,320.15,325.15,330.15,335.15,340.15,345.15,355.15,360.15,365.15,370.15,375.15,380.15,385.15,390.15,395.15,400.15,405.15,415.15,420.15,425.15,430.15,435.15,440.15,445.15,450.15,455.15,460.15,465.15,470.15,475.15,480.15,485.15,495.15,500.15,505.15,510.15,515.15,520.15,525.15,530.15,535.15,540.15,545.15,550.15,555.15,560.15,565.15,570.15,575.15,580.15,585.15,590.15,595.15]
unsampled_P = []

# Define LAMMPS log file names
lammps_log = "_log.lammps" # name of files starting with a sequential number (e.g. 1_log.lammps, 2_log.lammps...)
#=============================================================================================
# PATHS (CHANGE ACCORDINGLY)
#=============================================================================================
# Define paths
directory = ""
processed_data = directory + 'processed_data/'
original_data = directory + 'log_files/'
temp = directory + 'temp/'

if os.path.exists(temp):
    shutil.rmtree(temp)
    os.mkdir(temp)
else:
    os.mkdir(temp)

if os.path.exists(processed_data):
    shutil.rmtree(processed_data)
    os.mkdir(processed_data)
else:
    os.mkdir(processed_data)
#=============================================================================================
# MAIN
#=============================================================================================
# Seeking for the first file
first_file = 0
while True:
    first_file = first_file + 1
    if os.path.exists(original_data + str(first_file) + lammps_log):
        break
first_file = str(first_file)

# Defining the limits to extract LAMMPS file data
for n,line in enumerate(open(original_data + first_file + lammps_log,'r')):
    if "Step " in line:
        rm_up = n+1

# Identify columns
head = open(original_data + first_file + lammps_log,'r').readlines()[rm_up-1].split()
for j in range(len(head)):
    if head[j] == 'PotEng':
        PotEng_ind = j
    if head[j] == 'E_mol':
        E_mol_ind = j
    if head[j] == 'Volume':
        Volume_ind = j
    if head[j] == 'Press':
        Press_ind = j

for m,line in enumerate(open(original_data + first_file + lammps_log,'r')):
    if "Loop time" in line:
        rm_dw = m
for l,line in enumerate(open(original_data + first_file + lammps_log,'r')):
    if "Nominal values" in line:
        nom = l
    
# Extract data from original LAMMPS log files
files = len(os.walk(original_data).next()[2]) # Number of files with results to be used

temperature = zeros([files], float64)

if ensemble == 'NPT':
    pressure = zeros([files], float64)
if ensemble == 'NVT':   
    volume = zeros([files], float64) 

print "Creating temporary files..."
i = 0
for ind in range(files):
    while True:
        i = i + 1
        if os.path.exists(original_data + str(i) + lammps_log):
            break
    filename = os.path.join(original_data + str(i) + lammps_log)
    lines = open(filename).readlines()
    open(temp + str(ind) + '.lammps', 'w').writelines(lines[rm_up:rm_dw])
    temperature[ind] = float(lines[nom+2])
    if ensemble == 'NPT':
        pressure[ind] = float(lines[nom+4])
    if ensemble == 'NVT':
        volume[ind] = float(lines[nom+4])

# Define arrays
filename = os.path.join(temp, '0.lammps')
T_max = int(commands.getoutput('wc -l %s' % filename).split()[0])
files = len(os.walk(temp).next()[2]) # Number of files with results to be used

Uconf = zeros([files,T_max], float64)

if ensemble == 'NPT':
    volume = zeros([files,T_max], float64)
if ensemble == 'NVT':   
    pressure = zeros([files,T_max], float64)


print "Reading data..."
for ind in range(files):
    # Collect data from files
    filename = os.path.join(temp + str(ind) + '.lammps')
    infile = open(filename, 'r')
    elements = infile.readline().split()
    K = len(elements)
            
    # Determine maximum number of snapshots
    filename = os.path.join(temp + str(ind) + '.lammps')
            
    # Allocate storage for original energies
    T_k = zeros([K], int32) # T_k[k] is the number of snapshots from umbrella simulation k
    x_kt = zeros([K,T_max], float64) # x_kt[k,t] is the position of snapshot t from energy k (in kcal/mol)
            
    # Read the energies.
    filename = os.path.join(temp + str(ind) + '.lammps')
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
            
    for line in lines:
        elements = line.split()
        for k in range(K):
            t = T_k[k]
            x_kt[k,t] = float(elements[k])
            T_k[k] += 1

    for k in range(T_max):
        Uconf[ind,k] = x_kt[PotEng_ind,k] - x_kt[E_mol_ind,k] # Calculates the Configurational Internal Energy (PotEn - Emol)
        if ensemble == 'NPT':
            volume[ind,k] = x_kt[Volume_ind,k]
        if ensemble == 'NVT':   
            pressure[ind,k] = x_kt[Press_ind,k]

if include_unsampled == 'yes':
   temperature = numpy.concatenate([temperature,unsampled_T])
   zeros_Uconf = zeros([len(unsampled_T),T_max], float64)
   Uconf = numpy.concatenate([Uconf,zeros_Uconf])
   if ensemble == 'NPT':
       zeros_volume = zeros([len(unsampled_T),T_max], float64)
       volume = numpy.concatenate([volume,zeros_volume])
       pressure = numpy.concatenate([pressure,unsampled_P])
   if ensemble == 'NVT':
       zeros_pressure = zeros([len(unsampled_T),T_max], float64)
       pressure = numpy.concatenate([pressure,zeros_pressure])
       unsampled_V = zeros([len(unsampled_T)], float64)
       for i in range(len(unsampled_T)):
           unsampled_V[i] = volume[0]
       volume = numpy.concatenate([volume,unsampled_V])

# Write temperatures.
filename = os.path.join(processed_data + 'temperature.dat')
print "Writing Temperature to '%s'..." % filename
outfile = open(filename, 'w')
for k in range(0,len(temperature)):
    outfile.write('%.3f' % temperature[k])
    outfile.write(' ')
outfile.write('\n')
outfile.close()

# Write Uconf.
filename = os.path.join(processed_data + 'uconf.dat')
print "Writing Uconf to '%s'..." % filename
outfile = open(filename, 'w')
for t in range(T_max):
    for k in range(0,len(temperature)):
        outfile.write('%.6f' % Uconf[k,t])
        outfile.write(' ')
    outfile.write('\n')
outfile.close()

# Write pressures.
filename = os.path.join(processed_data + 'pressure.dat')
print "Writing Pressure to '%s'..." % filename
outfile = open(filename, 'w')
if ensemble == 'NPT': 
    for k in range(0,len(temperature)):
        outfile.write('%.3f' % pressure[k])
        outfile.write(' ')
    outfile.write('\n')
if ensemble == 'NVT': 
    for t in range(T_max):
        for k in range(0,len(temperature)):
            outfile.write('%.4f' %pressure[k,t])
            outfile.write(' ')
        outfile.write('\n')
outfile.close()

# Write volumes.
filename = os.path.join(processed_data + 'volume.dat')
print "Writing Volumes to '%s'..." % filename
outfile = open(filename, 'w')
if ensemble == 'NPT': 
    for t in range(T_max):
        for k in range(0,len(temperature)):
            outfile.write('%.4f' %volume[k,t])
            outfile.write(' ')
        outfile.write('\n')
if ensemble == 'NVT': 
    for k in range(0,len(temperature)):
        outfile.write('%.3f' % volume[k])
        outfile.write(' ')
    outfile.write('\n')
outfile.close()

shutil.rmtree(temp)

print "DONE!"
