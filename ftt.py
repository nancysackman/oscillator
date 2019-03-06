
import obspy
from obspy import read
from obspy.signal.tf_misfit import plot_tfr
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import numpy as np
import matplotlib #.pyplot as plt - for crash
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt #added for crash
#import matplotlib.pyplot as plt
from scipy.fftpack import fft

client = Client("IRIS")

t1 = UTCDateTime("2001-02-28T18:54:00.000")
t2 = t1 + 3600
st = client.get_waveforms("UW", "HOLY", "--", "ENN", t1, t2)
plt.figure(0)
st.plot(outfile = "quicktest.png")
tr = st[0]

N = tr.stats.npts
T = tr.stats.delta

x = tr.data
y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
plt.figure()
plt.loglog(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.plot(xf,2.0/N * np.abs(yf[0:int(N/2)])) #test this works as linear plot, if comment out loglog
plt.grid()
plt.show()
plt.savefig("quicktest2.png")

plt.figure(3)
plot_tfr(tr.data, dt=tr.stats.delta, fmin=.01, fmax=50., w0=8., nf=64, fft_zero_pad_fac=4)
plt.savefig("quicktest3.png")
