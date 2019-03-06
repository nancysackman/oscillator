import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
import numpy as np
import sys


import csv #downloading csv for seismograph
import requests #access files of IRIS webservices (eventually)
import re #regular expression
import webbrowser
import requests

url='https://service.iris.edu/irisws/timeseries/1/query?'
sta=sys.argv[1]
chan=sys.argv[2]
net = sys.argv[3]
loc = sys.argv[4]
starttime = sys.argv[5]
endtime = sys.argv[6]
format = sys.argv[7]

params="&sta=" + sta + "&chan=" + chan + "&net=" + net + "&loc" + loc + "&starttime=" +starttime + "&endtime" + endtime + "&format" + format
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


F0=[] #seismogram
for row in building_reader:
    #F0.append(float(row[1]))
    # print(row)
    pass
F0=np.array(F0)#making into numpy array
arr_out = [] #array out

#variables
#F=-kx
#F=ma
#ma=-kx
#a/x=-k/m: (m/s^2)/m=-k/kg:  s^2=-k/kg:  s=-sqrt(k/kg)
#s=seconds which equals period T
m=2 #mass - kg
k=240 #spring constant - units - kg/s^2
fp=math.sqrt(k/m) # angular frequency - kg/(s^2*kg) - free period - hz - squareroot of k/m
cdc=2*math.sqrt(k*m) #critical damping coefficient 2*squareroot of k*m
ga=.05*cdc #damping - use 5% damping
omega=fp #angular frequency, 2*pi/T=2*pi*f

delta_t=.01 #1/sample rate

#initial conditions
y=np.array([0,0]) #[velocity, displacement]

A=np.array([[m,0],[0,1]])
B=np.array([[ga,k],[-1,0]])
F=np.array([0.0,0.0])
Y=[]
force=[]
t=0.0
time=[]
gain=421885.50 #sensitivity from IRIS page counts/m/s^2
F0=F0/gain
mean=np.mean(F0)
F0=F0-mean
 #this is acceleration in m/s^2
#this is the displacement calculation loop
for ForcingValue in F0:
    time.append(t)
    t=t+delta_t
    F[0]=ForcingValue#*np.cos(omega*t) #need to verify that displacement is calculated this way or by another function
    y=y+delta_t*inv(A).dot(F-B.dot(y))
    Y.append(y[1])
    force.append(F[0])
    KE=0.5*m*y[0]**2
    PE=0.5*k*y[1]**2
    energy=KE+PE

#plot results
#t1=[i for i in time]
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.1, 0.1])

ax1=plt.subplot()
ax1.plot(time,force)
plt.ylabel('Acceleration m/s^2')
ax2=ax1.twinx()
ax2.plot(time,Y,c='r',linewidth=3.0)
plt.xlabel('Time in Seconds')
plt.ylabel('Displacement in Meters')
#ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#plt.plot(om,force)
plt.grid(True)
plt.legend(['Displacement','Force'],loc='upper right')
plt.show()

print ('Damping',np.sqrt((-ga**2+4*m*k)/(2.0*m)))
print ('Natural Frequency:',np.sqrt(k/m))
