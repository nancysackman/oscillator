import math
from numpy.linalg import inv
import matplotlib #.pyplot as plt - for crash
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt #added for crash
import numpy as np
import sys

#for frequency domain
import obspy
from obspy import read
from obspy.signal.tf_misfit import plot_tfr
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from scipy.fftpack import fft

#import numpy.fft as fft


import csv #downloading csv for seismograph
import requests #access files of IRIS webservices (eventually)
import re #regular expression
import webbrowser
import requests

# The Original ... Simple Single Degree of Freedom Oscillator

#variables

s=.6 #number of stories
rp=s*0.1 #resonance period, 0.1*number of stories, free period
TL = 5410000 #(5410 kN for total load 1 story)
g=9.81 #acceleration due to gravity
m = (TL/g)*s #mass, units kg for 6 story 50mx50m
k = 6580000.0#spring constant for 6 story, units kg/s^2, for 60,000kg, k=21600.0, highK=6580000.0
#calculate k - 2*pi*f=sqrt(k/m)
#k = #((2*math.pi/rp)**2)*m


CD = 2*math.sqrt(m*k) #critical damping
c = .05*CD #actual damping, units kg/s

#Damping - We want a 5% damping ratio, .05=c/2*sqrt(m*k)
#CDE=np.sqrt((-c**2 + 4*m*k)/(2.0*m)) #sq((-c^2 + 4*m*k))2*m critical damping equation
#E=c/CD #ratio of damping coefficient to critical damping - Critical Damping Coefficient
#We want a 5% damping ratio, .05=c/2*sqrt(m*k)
#Notes - 2*E*NF=c

NF = math.sqrt(k/m) #natural frequency, units Hz
T=(2*math.pi)/NF #1/NF, units seconds
f=1/T
#f=1/rp #frequency, units Hz


#F0 = 1
delta_t = .01 #.005KHZ, #0.02HOM, AK; 0.01 HOLY, UW; .005 K223, AK
omega = NF*(2*math.pi)/T #units Hz, w=sqrt(k/m) or 2pi/T, rads/sec
#time = np.arange(0.0, 60.0, delta_t) This is for the new time stepping solution
#ton=10
#toff=ton+10000*delta_t

#Here we add the import part with IRIS

url='https://service.iris.edu/irisws/timeseries/1/query?'
"""sta=sys.argv[1]
chan=sys.argv[2]
net = sys.argv[3]
loc = sys.argv[4]
starttime = sys.argv[5]
endtime = sys.argv[6]
format = sys.argv[7]"""
print("Enter STA,CHAN,NET,LOC,START TIME, END TIME, FORMAT")
#sta=raw_input("Enter Station")
#chan=raw_input("Enter Channel")
#net = raw_input("Enter Network")
#loc = raw_input("Enter Location")
#starttime = raw_input("Enter Start Time")
#duration = raw_input("Duration in Seconds")
#format = raw_input("Enter Output Format")
sta="HOLY" #khz,pcdr
chan="ENN" #hnn,hnn
net = "UW" #NZ,PR
loc = "--" #20
starttime = "2001-02-28T18:54:00"
duration = "120"
format = "geocsv"
params="sta=" + sta + "&cha=" + chan + "&net=" + net + "&loc=" + loc + "&start=" +starttime + "&dur=" + duration + "&format=" + format
print(params)
url+=params
print(url)
# params={'net':'IU&','sta':'ANMO&','loc':'00&','cha':'BHZ&','starttime':'2005-01-01T00:00:00&','endtime':'2005-01-02T00:00:00&','output':'plot'}

r=requests.get(url=url)
print(r)
file = open("./test.csv", "w")
file.write(r.text)
file.close()
building_reader=csv.reader(open("/Users/nancysackman/Code/building/test.csv"))

#This part is to import the csv file and skip headers from the IRIS website

for header in building_reader: #skip header information until reached first row of meaningful data
    if (header[0]!="Time"):
        next(building_reader)
    else:
        break

F0=[] #seismogram F=m*a,
for row in building_reader:
    F0.append(float(row[1]))
F0=np.array(F0)#making into numpy array
arr_out = [] #array out

#find frequency of accelerogram - actually need frequency of displacement

#data = np.array(F0)
#spectrum = fft.fft(data)
#freq = fft.fftfreq(len(spectrum))
#plt.plot(freq, abs(spectrum))
#plt.show()

#Now we come back to the oscillator program after variables have been selected

#initial state
y=np.array([0,0]) #[velocity, Displacement] y is a vector

A = np.array([[m,0],[0,1]]) #matrix, list of two lists, first row is m, second row is 0 and 1
B = np.array([[c,k], [-1,0]]) #damping and spring constant
F = np.array([0.0, 0.0]) #forcing vector
Y = [] #for plotting
force = []
acceleration = []

#This part had to be inserted to account for the gain of the accelerometer

t=0.0
#F[0] = F0 * np.sin(omega*t)
time = []
gain = 428155.0# 427336 KHZ,#523574.0 PCDR, #419102.0 sensitivity from IRIS page counts/m/s^2 - HOM, AK; 428155.0, HOLY, UW
F0 = F0/gain #F=m*a,
mean = np.mean(F0)
F0 = F0-mean


#Now we go back to the original part of the oscillator

#time stepping solution - original

#for t in time:
    #if t <= 0.0: #ton:
        #F[0] = F0*np.sin(omega*t)
        #F[0] = 0.0

    #else:
        #F[0] = 0.0
        #if t > 0.0: #<= toff:
            #F[0] = F0*np.sin(omega*t)
#here is where I am changing the code
        #if t > toff:
            #F[0]=0.0

#time stepping solution for seismogram

for ForcingValue in F0:
    time.append(t)
    t=t+delta_t
    F[0]=ForcingValue#*np.sin(omega*t) #need to verify that displacement is calculated this way or by another function

    y = y+delta_t*inv(A).dot(F*m-B.dot(y)) #F=m*a, since forcing function was acceleration, need to convert to force
    Y.append(y[1])
    #force.append(F[0])#
    acceleration.append(F[0])
    KE = 0.5*m*y[0]**2
    PE = 0.5*k*y[1]**2

    #if t % 1<=0.01:
    #    print'Total Energy:', KE+PE

#Plot results - this is a new version with labels

print('maximum displacement in meters',max(Y))
print('minimum displacement in meters',min(Y))

print('max displacement occurs at',Y.index(max(Y))*delta_t,'seconds')
print('min displacement occurs at',Y.index(min(Y))*delta_t,'seconds')

t=[i for i in time]
ax1=plt.subplot()
ax1.plot(time,acceleration)
plt.ylabel('Acceleration m/s^2')
plt.legend(['Force'],loc='upper right')
ax2=ax1.twinx()
ax2.plot(time,Y,c='r',linewidth=3.0)
plt.xlabel('Time in Seconds')
plt.ylabel('Displacement in Meters')

plt.grid(True)
plt.legend(['Displacement'],loc='upper center')

plt.show()


#old plot results
#t = [i for i in time]
#plt.plot(t,Y)
#plt.plot(t,force)
#plt.grid(True)
#plt.legend(['Displacement','Force'], loc='lower right')

#plt.show()

#print 'Critical Damping:', np.sqrt((-c**2 + 4*m*k)/(2.0*m)) #sq((-c^2 + 4*m*k))2*m
#print 'Natural Frequency:',np.sqrt(k/m)
print(m,k,c,CD,NF,rp,T,f,omega)

#The next part is going to look at building displacement based on the number of stories,
#mass, spring constant, critical damping ratio.  We need to equate or figure out
#the frequency of the acceleration or forcing function from the seismogramself.
#To do this I have equated the masses from F=ma and omega = sqrt(k/m)=2pifself.
#If I set the equations equal to each other based on m, then f=sqrt(F*k/4pi^2*a)

#new code called frequency
#a is acceleration from spectogram

#f=sqrt(F*k/4*math.pi**2*a)
