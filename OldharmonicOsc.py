#!/usr/bin/env python
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

import csv #downloading csv for seismograph
import requests #access files of IRIS webservices (eventually)
import re #regular expression

#building_reader=csv.reader(open("/Users/nancysackman/Code/UW.HOLY.ENZ.2001.csv"))
#from itertools import islice
#next(islice(building_reader, 19, 19), None) #skips 19 rows of headers
#for row in building_reader:
    #col1_str=re.search('# .*', row[0])
    #if col1_str:
        #col1=re.search('# .*', row[0])
        #print(col1_str)
    # if col1==('#'):
    #     next(building_reader,None)


#2001-02-28T18:54:44.000000Z
#'-?\d.*\.\d*'
#\d	Any digit
#\D	Any non-digit
#.	Any single character
#a*	Zero or more of a
# for FF in count:
#     count[:,1]
#     print(FF)
#put count of 1 column 2, into the forcing function

#print(building_reader.next())
arr_out = []

#make a loop to extract time stamp

#for timeStamp in building_reader:
#    if timeStamp=re.search('2001-02-28T')



#variables
m=2 #mass - kg 2
k=8 #spring constant - units - kg/s^2 8
fp=math.sqrt(k/m) # real space - kg/(s^2*kg) - free period - hz - squareroot of k/m
cdc=2*math.sqrt(k*m) #critical damping coefficient 2*squareroot of k*m
ga=cdc*.05 #damping - use 5% damping
omega=fp #sinusoidal frequency, 2*pi/T=2*pi*f
T=1/fp
print(m,k,cdc,ga,fp,T)

F0=7 #forcing at time 0
F1=5 #forcing at time later
delta_t=.001
fon=10 #time in seconds to turn on forcing
foff=fon+1  #time in seconds to turn off forcing, fon+30, or fon+ 10000 * delta_t
time=np.arange(0.0,60.0,delta_t)
#om=np.arange(0,10*math.pi, math.pi/2)

#initial conditions
y=np.array([0,0]) #[velocity, displacement]

A=np.array([[m,0],[0,1]])
B=np.array([[ga,k],[-1,0]])
F=np.array([0.0,0.0])
Y=[]
force=[]

#time step solution
for t in time:
    if t<=fon: #this is to turn on forcing - new code
        F[0]=0.0
    # if t<=20: this is Srividha's code
    #     F[0]=F0*np.cos(omega*t) - Srividhya's code
        #F[1]=F1*np.cos(omega*t) this is original code
    #else:

        #F[0]=F1*np.cos(omega*t)
    else:
        if t<=foff:
            F[0] = F0 * np.cos(omega*t)
        #F[0]=F1*np.cos(omega*t) also Srividha's code
        #F[0]=0.0
    if t>=foff:
        F[0]=0.0

    y=y+delta_t*inv(A).dot(F-B.dot(y))
    Y.append(y[1])
    force.append(F[0])
    #force.append(F[1])

    KE=0.5*m*y[0]**2
    PE=0.5*k*y[1]**2
    energy=KE+PE

    if t % 1<=0.01:
        print ('Total Energy:',KE+PE)


#plot results
t=[i for i in time]
plt.plot(t,Y)
plt.plot(t,force)
#plt.plot(om,force)
plt.grid(True)
plt.legend(['Displacement','Force'],loc='lower right')
plt.show()

print ('Damping',np.sqrt((-ga**2+4*m*k)/(2.0*m)))
print ('Natural Frequency:',np.sqrt(k/m))
