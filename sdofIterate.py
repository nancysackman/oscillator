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

#Here we add the import part with IRIS

url='https://service.iris.edu/irisws/timeseries/1/query?'
sta="SSN" #khz,pcdr TLIG
chan="BNN" #hnn,hnn BHN
net = "AK" #NZ,PR MX
loc = "--" #20
starttime = "2018-11-30T17:29:28"
duration = "100"
format = "geocsv"
params="sta=" + sta + "&cha=" + chan + "&net=" + net + "&loc=" + loc + "&start=" +starttime + "&dur=" + duration + "&format=" + format
print(params)
url+=params
print(url)
#params={'net':'IU&',/Users/nancysackman/Code/building/sdofIterate.py'sta':'ANMO&','loc':'00&','cha':'BHZ&','starttime':'2005-01-01T00:00:00&','endtime':'2005-01-02T00:00:00&','output':'plot'}

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

#This part had to be inserted to account for the gain of the accelerometer

gain = 427894.0# 427336 KHZ,#523574.0 PCDR, #419102.0 sensitivity from IRIS page counts/m/s^2 - HOM, AK; 428155.0, HOLY, UW, K211 213947.0
F0 = F0/gain #F=m*a,
mean = np.mean(F0) #gets the mean for all F0(accelerogram)
F0 = F0-mean #the mean is then subtracted after dividing gain
#variables

for s in range (1,5):
    #s=1 #number of stories
    rp=s*0.1 #resonance period, 0.1*number of stories, free period
    g=9.81 #acceleration due to gravity
    m = 5410000*s/g #(TL/g)*s ,5410000, #mass, units kg for 6 story 50mx50m
    k = 11700000 #spring constant
    #Damping - We want a 5% damping ratio, .05=c/2*sqrt(m*k)
    #CDE=np.sqrt((-c**2 + 4*m*k)/(2.0*m)) #sq((-c^2 + 4*m*k))2*m critical damping equation
    NF = math.sqrt(k/m) #natural frequency, units Hz
    T=(2*math.pi)/NF #1/NF, units seconds
    f=1/T
    delta_t = .02 #.005KHZ, #0.02HOM, AK; 0.01 HOLY, UW; .005 K223, AK
    omega = NF*(2*math.pi)/T #units Hz, w=sqrt(k/m) or 2pi/T, rads/sec
    CD = 2*math.sqrt(m*k) #critical damping
    c = .05*CD #actual damping, units kg/s

    #initial state
    y=np.array([0,0]) #[velocity, Displacement] y is a vector
    A = np.array([[m,0],[0,1]]) #matrix, list of two lists, first row is m, second row is 0 and 1
    B = np.array([[c,k], [-1,0]]) #damping and spring constant
    F = np.array([0.0, 0.0]) #force vector
    Y = [] #for plotting
    force = []
    acceleration = []

    #This part had to be inserted to account for the gain of the accelerometer
    t=0.0
    time = []
    #This part is to create the Displacement

    for ForcingValue in F0:
        time.append(t)
        t=t+delta_t
        F[0]=ForcingValue#*np.sin(omega*t) #need to verify that displacement is calculated this way or by another function

        y = y+delta_t*inv(A).dot(F*m-B.dot(y)) #F=m*a, multiply-acceleration coming in, need to convert to force
        Y.append(y[1])
        #force.append(F[0])#
        acceleration.append(F[0])
        KE = 0.5*m*y[0]**2
        PE = 0.5*k*y[1]**2

    #plot sdofResults

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
    plt.title('Building Displacement per Story'+str(s))
    plt.show()

    #print stuff

    print('absolute max acceleration',max((acceleration),key=abs))
    print('max acceleration occurs at',acceleration.index(max((acceleration),key=abs))*delta_t,'seconds')
    print('maximum displacement in meters',max((Y),key=abs))
    print('max displacement occurs at',Y.index(max((Y),key=abs))*delta_t,'seconds')


    #Output files - loop following
        #for s in range (1,31):

    arrSDOF=[sta,chan,net,gain,s,m,NF,f,delta_t,T,max((acceleration),key=abs), acceleration.index(max(acceleration))*delta_t, max((Y),key=abs), Y.index(max((Y),key=abs))*delta_t]
    with open('sdofResultspython.csv',mode='a') as csv_out:
        writer=csv.writer(csv_out)
        writer.writerow(arrSDOF)
