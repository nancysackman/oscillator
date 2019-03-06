import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

import csv #downloading csv for seismograph
#simport requests #access files of IRIS webservices (eventually)
import re #regular expression

building_reader=csv.reader(open("/Users/nancysackman/Code/K213AKtest.csv"))
#while row[0] and while row[1]=time and sample
for header in building_reader: #skip header information until reached first row of meaningful data
    if (header[0]!="Time"):
        next(building_reader)
    else:
        break

F0=[] #seismogram
for row in building_reader:
    F0.append(float(row[1])) #row of 1 means second column, row 0 is first column
F0=np.array(F0)#making into numpy array
arr_out = [] #array out

#variables
#F=-kx
#F=ma
#ma=-kx
#a/x=-k/m: (m/s^2)/m=-k/kg:  s^2=-k/kg:  s=-sqrt(k/kg)
#s=seconds which equals period T
#mu"+ju'+ku=FF - partial differential equation, m=mass, j=damping, k=spring constant, u=displacement
m=60000 #mass - kg
k=21600 #spring constant - units - kg/s^2 calculated with 6 stories=.6s
fp=math.sqrt(k/m) # angular frequency - kg/(s^2*kg) - free period - hz - squareroot of k/m
cdc=2*math.sqrt(k*m) #critical damping coefficient 2*squareroot of k*m
ga=.05*cdc #damping coefficient - use 5% damping ratio
#damping ratio is E-damping coefficient c/critical damping cc
omega=fp #angular frequency, 2*pi/T=2*pi*f

delta_t=.005 #1/sample rate

#initial conditions
y=np.array([0,0]) #[velocity, displacement]

A=np.array([[m,0],[0,1]]) # list of list, first row is m, second row is 1
B=np.array([[cdc,k],[-1,0]]) #list of list, changing to cdc from ga
F=np.array([0.0,0.0])
Y=[]
force=[]
t=0.0
time=[]
gain=213947.0 #sensitivity from IRIS page counts/m/s^2
F0=F0/gain
mean=np.mean(F0)
F0=F0-mean
 #this is acceleration in m/s^2
#this is the displacement calculation loop
for ForcingValue in F0:
    time.append(t)
    t=t+delta_t
    F[0]=ForcingValue#*np.cos(omega*t) #need to verify that displacement is calculated this way or by another function
    y=y+delta_t*inv(A).dot(F-B.dot(y))
    Y.append(y[1])
    force.append(F[0])
    KE=0.5*m*y[0]**2
    PE=0.5*k*y[1]**2
    energy=KE+PE
    #print(energy)

#plot results
#t1=[i for i in time]
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.1, 0.1])

ax1=plt.subplot()
ax1.plot(time,force)
plt.ylabel('Acceleration m/s^2')
plt.legend(['Force'],loc='upper right')
ax2=ax1.twinx()
ax2.plot(time,Y,c='r',linewidth=3.0)
plt.xlabel('Time in Seconds')
plt.ylabel('Displacement in Meters')
#ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plt.plot(om,force)
plt.grid(True)
plt.legend(['Displacement'],loc='upper center')

plt.show()

print ('Critical Damping:',np.sqrt((-cdc**2+4*m*k)/(2.0*m))) #changing from ga to cdc
print ('Natural Frequency:',np.sqrt(k/m))
