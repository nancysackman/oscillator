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

building_reader=csv.reader(open("/Users/nancysackman/Code/building/MI1520170919181440.csv"))

#manually fill in from txt files
sta="MI15" #khz,pcdr TLIG
chan="S00E" #hnn,hnn BHN
net = "UNAM" #NZ,PR MX
starttime = "2017-09-19T18:14:40"
samplingtime="2017-09-19T18:15:00" #start time of instrument recording
depth="57"
M="7.1"
gain=0

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
F0=F0/100 #change to cm for Mexico
arr_out = [] #array out

#variables

for s in range (1,2): #s is number of stories

    rp=s*0.1 #resonance period, 0.1*number of stories, free period
    g=9.81 #acceleration due to gravity
    m = 5410000*s/g #(TL/g)*s ,5410000, #mass, units kg for 6 story 50mx50m
    k = 11700000 #spring constant,k=9.8MN/m which is 9800000, k=AE/L 11700000, k reinforced concrete 100000000
    CD = 2*math.sqrt(m*k) #critical damping
    c = .05*CD #actual damping, units kg/s

    #Damping - We want a 5% damping ratio, .05=c/2*sqrt(m*k)
    #CDE=np.sqrt((-c**2 + 4*m*k)/(2.0*m)) #sq((-c^2 + 4*m*k))2*m critical damping equation
    #E=c/CD #ratio of damping coefficient to critical damping - Critical Damping Coefficient

    NF = math.sqrt(k/m) #natural frequency, units Hz
    T=(2*math.pi)/NF #1/NF, units seconds
    f=1/T
    delta_t = .01 #.005KHZ, #0.02HOM, AK; 0.01 HOLY, UW; .005 K223, AK
    omega = NF*(2*math.pi)/T #units Hz, w=sqrt(k/m) or 2pi/T, rads/sec

    #Now we come back to the oscillator program after variables have been selected

    #initial state
    y=np.array([0,0]) #[velocity, Displacement] y is a
    A = np.array([[m,0],[0,1]]) #matrix, list of two lists, first row is m, second row is 0 and 1
    B = np.array([[c,k], [-1,0]]) #damping and spring constant
    F = np.array([0.0, 0.0]) #forcing vector
    Y = [] #for plotting
    force = []
    acceleration = []

    t=0.0 #genius! there is no lagtime
    time = []

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

    t=[i for i in time]
    ax1=plt.subplot()
    ax1.plot(time,acceleration)
    plt.xlim((0,175))
    plt.ylabel('Acceleration m/s^2')
    plt.xlabel('Time in Seconds')
    plt.legend(['Force'],loc='upper right')
    ax2=ax1.twinx()
    ax2.plot(time,Y,c='r',linewidth=3.0)
    plt.xlabel('Time in Seconds')
    plt.ylabel('Displacement in Meters')
    plt.grid(True)
    plt.legend(['Displacement'],loc='upper center')
    plt.title('Building Displacement - Story '+ str(s)+' '+ str(sta)+ ' '+ str(chan)+ ' '+ str(starttime))
    plt.show()

    #print stuff

    print('absolute max acceleration',max((acceleration),key=abs))
    print('max acceleration occurs at',acceleration.index(max((acceleration),key=abs))*delta_t,'seconds')
    print('maximum displacement in meters',max((Y),key=abs))
    print('max displacement occurs at',Y.index(max((Y),key=abs))*delta_t,'seconds')

    #send out to CSV file
    arrSDOF=[sta,chan,net,delta_t,starttime,samplingtime,gain,s,rp,m,k,NF,T,f,KE,PE,max((acceleration),key=abs), acceleration.index(max(acceleration))*delta_t, max((Y),key=abs), Y.index(max((Y),key=abs))*delta_t]
    with open('sdofResultspython.csv',mode='a') as csv_out:
        writer=csv.writer(csv_out)
        writer.writerow(arrSDOF)

#The next part is going to look at building displacement based on the number of stories,
#mass, spring constant, critical damping ratio.  We need to equate or figure out
#the frequency of the acceleration or forcing function from the seismogramself.
#To do this I have equated the masses from F=ma and omega = sqrt(k/m)=2pifself.
#If I set the equations equal to each other based on m, then f=sqrt(F*k/4pi^2*a)

#new code called frequency
#a is acceleration from spectogram

#f=sqrt(F*k/4*math.pi**2*a)
