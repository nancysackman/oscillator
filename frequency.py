#This part is going to look at building displacement based on the number of stories,
#mass, spring constant, critical damping ratio.  We need to equate or figure out
#the frequency of the acceleration or forcing function from the seismogramself.
#To do this I have equated the masses from F=ma and omega = sqrt(k/m)=2pifself.
#If I set the equations equal to each other based on m, then f=sqrt(F*k/4pi^2*a)


import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.integrate as it

import csv #downloading csv for seismograph
import requests #access files of IRIS webservices (eventually)
import re #regular expression
import webbrowser
import requests

s=1 #number of stories
rp=s*0.1 #resonance period, 0.1*number of stories, free period
TL = 5410000 #(5410 kN for total load 1 story)
g=9.81 #acceleration due to gravity
m = (TL/g)*s #mass, units kg for 6 story 50mx50m
k = 6580000.0#spring constant for 6 story, units kg/s^2, for 60,000kg, k=21600.0, highK=6580000.0
CD = 2*math.sqrt(m*k) #critical damping
c = .05*CD #actual damping, units kg/s

#Damping - We want a 5% damping ratio, .05=c/2*sqrt(m*k)
#CDE=np.sqrt((-c**2 + 4*m*k)/(2.0*m)) #sq((-c^2 + 4*m*k))2*m critical damping equation
#E=c/CD #ratio of damping coefficient to critical damping - Critical Damping Coefficient
#We want a 5% damping ratio, .05=c/2*sqrt(m*k)
#Notes - 2*E*NF=c
NF = math.sqrt(k/m) #natural frequency, units Hz (?), rads/sec
T=(2*math.pi)/NF #1/NF, units seconds
delta_t = 0.01 #0.02HOM, AK; 0.01 HOLY, UW
omega = NF #units Hz, w=sqrt(k/m) or 2pi/T, rads/sec

url='https://service.iris.edu/irisws/timeseries/1/query?'
"""sta=sys.argv[1]
chan=sys.argv[2]
net = sys.argv[3]
loc = sys.argv[4]
starttime = sys.argv[5]
endtime = sys.argv[6]
format = sys.argv[7]"""
print("Enter STA,CHAN,NET,LOC,START TIME, END TIME, FORMAT")
sta="HOLY"
chan="ENN"
net = "UW"
loc = "--"
starttime = "2001-02-28T18:54:00"
duration = "120"
format = "geocsv"
params="sta=" + sta + "&cha=" + chan + "&net=" + net + "&loc=" + loc + "&start=" +starttime + "&dur=" + duration + "&format=" + format
print(params)
url+=params
print(url)

r=requests.get(url=url)
print(r)
file = open("./test.csv", "w")
file.write(r.text)
file.close()
building_reader=csv.reader(open("/Users/nancysackman/Code/building/test.csv"))


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
print(F0)

#initial state
y=np.array([0,0]) #[velocity, Displacement] y is a vector

A = np.array([[m,0],[0,1]]) #matrix, list of two lists, first row is m, second row is 0 and 1
B = np.array([[c,k], [-1,0]]) #damping and spring constant
F = np.array([0.0, 0.0]) #forcing vector
Y = [] #for plotting
force = []
acceleration = []

#a is acceleration from spectogram

#f=sqrt(F*k/4*math.pi**2*a)

t=0.0
#F[0] = F0 * np.sin(omega*t)
time = []
gain = 428155.0 #419102.0 sensitivity from IRIS page counts/m/s^2 - HOM, AK; 428155.0, HOLY, UW
F0 = F0/gain #F=m*a,
mean = np.mean(F0)
F0 = F0-mean

v=[0]
x=[0]

for acceleration in F0:
    time.append(t)
    t=t+delta_t
    F[0]=acceleration
    #print(F[0])
    v.append(v[-1]+acceleration*t)
    #x.append(x[-1]+0.5*acceleration*t**2)
    print(v)
    print(x)

#for a in acceleration:

#velocity = it.cumtrapz(F[0],initial=0)
#printe(velocity)
