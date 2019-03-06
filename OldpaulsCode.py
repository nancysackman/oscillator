#!/usr/bin/env python

import math
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

#variables

# In the case of a cantilever beam,
# the "spring constant" is a function of Young's Modulus
# the Area Moment of Intertia, and the Length....something like
#3*E*I/l^3

# examples from http://civilengineer.webinfolist.com/str/micalcr.php

#YM = 3E10 # concrete = 30 GPa; GPa = kg/m/s^2 X 10^9
#AMI = 0.225 # example: 0.1m X 3m high= .225 m^4
#Length = 3 # one story structure ~ 3m?f
#k = (3 * YM * AMI) / Length**3
#m = 0.03 * 3600 # for a .1 X .1 X 3 m pillar with density 3600 kg/m^3

# Trying to express damping as a % of critical
#DP = 0.05 # desired Percentage of critical (i.e., .05 = 5% DC)
#CD = 2 * math.sqrt(m * k)
#c = .95 * CD

#print(YM, AMI, Length, k, m, DP, CD, c)

# The Original ...
m = 1.0
k = 2.0
CD = 2 * math.sqrt(m * k)
c = .05 * CD
FP = math.sqrt(k/m)
T=1/FP
print(m,k,CD,c,FP,T)
#c = 0.40 #critical damping = 2 * SQRT(m*k) = 4.0




F0 = 1.0
delta_t = 0.001

ton = 10.0 #time to turn on forcing
toff = ton + 10000 * delta_t #impulse
#toff = 20.0 #time to cut off forcing

#Here's a tricky one...driving frequency = free vibration frequency
omega = FP
#delta_t = 1
#omega = 1.0
time = np.arange(0.0, 60.0, delta_t)

#initial state
y = np.array([0,0]) #[velocity, displacement]

A = np.array([[m,0],[0,1]])
B = np.array([[c,k],[-1,0]])
F = np.array([0.0,0.0])

Y = []
force = []
energy = []

#time-stepping solution
for t in time:
    if t <= ton:
        F[0] = 0.0
#F[0] = F0 * np.cos(omega*t) # harmonic forcing, starts at +1

    else:
        if t<= toff:
            F[0] = F0 * np.sin(omega*t) # harmonic forcing, builds up from zero
# F[0] = 1000 #impulse!

    if t>toff:
        F[0] = 0.0

    y = y + delta_t * inv(A).dot( F - B.dot(y))
    Y.append(y[1])
    force.append(F[0])

    KE = 0.5 * m *y[0]**2
    PE = 0.5 * k *y[1]**2

    energy = KE + PE
        # print (energy)
        # energy.append(E[0])

    if t % 1 <= 0.01:
        print(t,energy,y[1])
            # print ('Total Energy:', KE+PE)

#plot the result
t = [i for i in time]
plt.plot(t,Y)
plt.plot(t,force)

#plt.plot(t,energy)
plt.grid(True)
plt.legend(['Force', 'Displacement'], loc='lower right')

plt.show()
print(m,k,CD,c,FP,T)

#print ('Critical Damping:' , np.sqrt((-c**2 + 4*m*k)/(2*m) )
#print ('Natural Frequency:' , np.sqrt(k/m) )
