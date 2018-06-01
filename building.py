#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import numpy as np
print('hello world')
#how to start this script in the terminal server or shell :

#.....cd Code\building\python building.py
#....then hit enter
#this script is going to be about the timing of building failure during an earthquake ...
#or seismic event

#Step 1 - Motion from seismograph - some time (t)
#Step 2 - Single degree oscillation - so linear, omega=sqrt(k/m), F=ma; spring - 5% damping - F=-kx; ...
#Step 2 - continued - Tn/L - change the natural period; period T=1/f; frequency f=1/T
#Step 3 - more equations, omega=2pif=2pi/T so T=2pi/sqrt(k/m) or T=2pi/omega
#Step 3 - continued, Dmax=Sa/omega^2, where D=distance, Sa=s-wave ampflication

#More notes -
#f'c=24.5Mpa , Mpa=F/A, f=ma, 9.81 m/s^2,
# m=


#find the force given a mass and acceleration F=ma
def find_force(mass, acceleration):
    #do something here so don't need to define variable, evaluates rightside before left side
    return mass*acceleration


force=find_force(10, 9.8)
print(force)

#find oscillation from mass omega=sqrt(k/m)

def find_oscillation (springk, mass):

    oscillation=math.sqrt(springk/mass)
    return oscillation

oscillation=find_oscillation(6, 10)
print(oscillation)

#find spring find_force

def find_springForce (springk, distance):

    springForce=-springk*distance
    return springForce

springForce=find_springForce(6, 10)
print(springForce)

#define period - T=1/f where f=frequency

def find_period (frequency):

    period=1/frequency
    return period

period=find_period(5)
print(period)

#define frequency where f=1/T

def find_frequency (period):

    frequency=1/period
    return frequency

frequency=find_frequency(15)
print(frequency)

#define omega where omega=2pif=2pi/T so T=2pi/sqrt(k/m) or T=2pi/omega

def find_omega(frequency):

    omega=math.pi*frequency*2
    return omega

omega=find_omega(15)
print(omega)

#define dMax for S-wave Dmax=Sa/omega^2

def find_dMax(omega,Sa):

    dMax=Sa/omega**2
    return dMax

dMax=find_dMax(omega,.98)
print(dMax)

#start some plots - build an aray to test

mArrayc=[3.0,3.25,3.50,3.75,4.0] #this is a python list
print(mArrayc)
a=9.81
m=np.array(mArrayc) #this is a numpy array - this is what I want
f=m*a
print(f)
plt.scatter(mArrayc,f)
plt.xlabel('Mass')
plt.ylabel('Force')
plt.title('Force vs. Mass of Building')
#plt.figure(figsize=(10,12))
plt.show()
