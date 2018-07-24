#!/usr/bin/env python
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np

#variables
m=600
k=2
ga=2

F0=30
delta_t=.01
omega=1
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
    if t<=20:
        F[0]=F0*np.cos(omega*t)
    else:
        F[0]=0.0

    y=y+delta_t*inv(A).dot(F-B.dot(y))
    Y.append(y[1])
    force.append(F[0])

    KE=0.5*m*y[0]**2
    PE=0.5*k*y[1]**2

    #if t % 1<=0.01:
    #    print ('Total Energy:',KE+PE)


#plot results
t=[i for i in time]
plt.plot(t,Y)
plt.plot(t,force)
plt.grid(True)
plt.legend(['Displacement','Force'],loc='lower right')
plt.show()

print ('Damping',np.sqrt((-ga**2+4*m*k)/(2.0*m)))
print ('Natural Frequency:',np.sqrt(k/m))
