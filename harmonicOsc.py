#!/usr/bin/env python
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

#variables
m=100 #mass - kg
k=100 #spring constant - units - kg/s^2
fp=math.sqrt(k/m) # angular frequency - kg/(s^2*kg) - free period - hz - squareroot of k/m
cdc=2*math.sqrt(k*m) #critical damping coefficient 2*squareroot of k*m
ga=.05*cdc #damping - use 5% damping
omega=fp #angular frequency, 2*pi/T=2*pi*f

F0=1 #forcing at time 0
F1=5 #forcing at time later
delta_t=.001
fon=10 #time in seconds to turn on forcing
foff=fon+20  #time in seconds to turn off forcing, fon+30, or fon+ 10000 * delta_t
time=np.arange(0.0,60.0,delta_t)

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
        F[0]=0
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
        F[0]=0

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
plt.grid(True)
plt.legend(['Displacement','Force'],loc='lower right')
plt.show()

print ('Damping',np.sqrt((-ga**2+4*m*k)/(2.0*m)))
print ('Natural Frequency:',np.sqrt(k/m))
