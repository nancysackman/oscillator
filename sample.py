import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
import csv #downloading csv for seismograph
import requests #access files of IRIS webservices (eventually)
import re #regular expression

building_reader=csv.reader(open("/Users/nancysackman/Code/UW.HOLY.ENZ.2001test.csv"))
F0=[] #seismogram
for row in building_reader:
    F0.append(float(row[1]))

F0=np.array(F0)#making into numpy array
arr_out = [] #array out

m=2 #mass - kg
k=100 #spring constant - units - kg/s^2
fp=math.sqrt(k/m) # angular frequency - kg/(s^2*kg) - free period - hz - squareroot of k/m
cdc=2*math.sqrt(k*m) #critical damping coefficient 2*squareroot of k*m
ga=.01*cdc #damping - use 5% damping
omega=fp #angular frequency, 2*pi/T=2*pi*f
delta_t=.001 #1/sample rate
#initial conditions
y=np.array([0,0]) #[velocity, displacement]

A=np.array([[m,0],[0,1]])
B=np.array([[ga,k],[-1,0]])
F=np.array([0.0,0.0])
Y=[]
force=[]
t=0.0
time=[]
gain=421885.50 #sensitivity from IRIS page counts/m/s^2
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
#plot results
#t1=[i for i in time]
ax1=plt.subplot()
ax1.plot(time,force)
plt.ylabel('Acceleration m/s^2')
ax2=ax1.twinx()
ax2.plot(time,Y,c='r',linewidth=3.0)
plt.xlabel('Time in Seconds')
plt.ylabel('Displacement in Meters')

#plt.plot(om,force)
plt.grid(True)
plt.legend(['Force','Displacement'],loc='lower right')
plt.show()

print ('Damping',np.sqrt((-ga**2+4*m*k)/(2.0*m)))
print ('Natural Frequency:',np.sqrt(k/m))
