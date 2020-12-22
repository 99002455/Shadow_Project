from scipy import signal as sp
import numpy as np
import IPython.display as ipd
import librosa as lbr
import wave
import struct

# Define waveread
def wavread(sndfile):
    wf=wave.open(sndfile,'rb');
    nchan=wf.getnchannels();
    bytes=wf.getsampwidth();
    rate=wf.getframerate();
    length=wf.getnframes();
    print("Number of channels: ", nchan);
    print("Number of bytes per sample:", bytes);
    print("Sampling rate: ", rate);
    print("Number of samples:", length);
    data=wf.readframes(length);
    if bytes==2:
        shorts = (struct.unpack( 'h' * length, data ));
    else:
        shorts = (struct.unpack( 'B' * length, data ));
    wf.close;
    return shorts, rate;

# Load audio file
x, fs = wavread('./audio/fspeech.wav')

# Play audio
display(ipd.Audio(x, rate = fs ))

#additive zero mean white noise (for -2**15<x<+2**15):
y=x+0.1*(np.random.random(np.shape(x))-0.5)*2**15
display(ipd.Audio(y, rate = fs ))


#make x and y  matrices and transpose them into a column:
x=np.matrix(x).T
y=np.matrix(y).T

#we assume 10 coefficients for our Wiener filter. 
#10 to 12 is a good number for speech signals.
A = np.matrix(np.zeros((100000, 10)))
for m in range(100000):
    A[m,:] = y[m+np.arange(10)].T
#Our matrix has 100000 rows and 10 colums:
print (A.shape)

#Compute Wiener Filter:
import matplotlib.pyplot as plt
h=np.linalg.inv(A.T*A)*A.T*x[5:100000+5]
plt.figure(figsize=(8,6))
plt.plot(np.flipud(h)) 
plt.xlabel('Sample')
plt.ylabel('value')
plt.title('Impulse Response of Wiener Filter')
plt.grid()

#plot
w,Hspeech=sp.freqz(x);
w,Hnoise=sp.freqz(0.1*(np.random.random(np.shape(x))-0.5)*2**15)
w,Hw=sp.freqz(np.flipud(h))
plt.figure(figsize=(8,6))
plt.plot(w,20*np.log10(abs(Hspeech))); 
plt.plot(w,20*np.log10(abs(Hnoise)),'r');
#plot and shift the filter into the vicinity of the signal:
plt.plot(w,20*np.log10(abs(Hw))+100,'g')
plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (dB)')
plt.legend(('Speech', 'White Noise', 'Wiener Filter'))
plt.title('Magnitude Spectrum')
plt.grid()

#Filtered Output
xw = sp.lfilter(np.array(np.flipud(h).T)[0],[1],np.array(y.T)[0])

display(ipd.Audio(xw, rate = fs ))

print( np.shape(x))
#Compute the quadratic error for the first 100000 samples:
print(sum(np.power(y[:100000]-x[:100000],2))/100000)

#MSE of filtered signal
print(np.sum(np.power(xw[4:100004]-x[:100000].T,2))/100000)




