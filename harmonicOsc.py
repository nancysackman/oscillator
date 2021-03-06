#!/usr/bin/env python
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

import csv #downloading csv for seismograph
import requests #access files of IRIS webservices (eventually)
import re #regular expression

building_reader=csv.reader(open("/Users/nancysackman/Code/UW.HOLY.ENZ.2001test.csv"))
#from itertools import islice
#next(islice(building_reader, 19, 19), None) #skips 19 rows of headers
F0=[]
#building forcing function from csv file
for row in building_reader:
    F0.append (row[1])

    # col1_str=re.search('# .*', row[0])
    # if col1_str:
    #     col1=next(building_reader,None)
    #     print(col1)
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
m=2 #mass - kg
k=8 #spring constant - units - kg/s^2
fp=math.sqrt(k/m) # angular frequency - kg/(s^2*kg) - free period - hz - squareroot of k/m
cdc=2*math.sqrt(k*m) #critical damping coefficient 2*squareroot of k*m
ga=.05*cdc #damping - use 5% damping
omega=fp #angular frequency, 2*pi/T=2*pi*f


# F0=1 #forcing at time 0
# F1=5 #forcing at time later
delta_t=.01
fon=10 #time in seconds to turn on forcing
foff=fon+20  #time in seconds to turn off forcing, fon+30, or fon+ 10000 * delta_t
time=np.arange(0.0,60.0,delta_t)
#om=np.arange(0,10*math.pi, math.pi/2)

#initial conditions
y=np.array([0,0]) #[velocity, displacement]

A=np.array([[m,0],[0,1]])
B=np.array([[ga,k],[-1,0]])
F=np.array([0.0,0.0])
Y=[]
force=[]
t=0.0
#time step solution
for ForcingValue in F0:
    t=t+delta_t
    print(t)
    #F[0]=ForcingValue*np.cos(omega*t)

    y=y+delta_t*inv(A).dot(F-B.dot(y))
    Y.append(y[1])
    force.append(F[0])
    #force.append(F[1])

    KE=0.5*m*y[0]**2
    PE=0.5*k*y[1]**2
    energy=KE+PE

    # if t % 1<=0.01:
    #     print ('Total Energy:',KE+PE)


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
