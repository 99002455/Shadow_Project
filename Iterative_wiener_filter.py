from scipy import signal as sp
from scipy import stats
import numpy as np
import IPython.display as ipd
import librosa as lbr
import wave
import struct
import matplotlib.pyplot as plt

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
clean_signal, fs = wavread('sp01.wav')
display(ipd.Audio(clean_signal, rate = fs ))
c = list(clean_signal)

# Load noisy file
noisy_signal, fs = wavread('sp01_train_sn10.wav')
display(ipd.Audio(noisy_signal, rate = fs ))
d = list(noisy_signal)

#make x and y  matrices and transpose them into a column:
clean_signal=np.matrix(clean_signal).T
noisy_signal=np.matrix(noisy_signal).T
print(noisy_signal.shape)
type(noisy_signal)


#number of samples for 20 milli_seconds is 0.020*sampling rate (rate in this case)
samples = int(0.020*8000)
print("no of samples for 20 milli sec:",samples)

# no of interations to be performed are (total no of samples/samples for 20 milli seconds)
iteration = int(22529/samples)
array = []
for i in range(1,iteration+1):
    k=0
    A = np.matrix(np.zeros((samples, 30)))
    #print(A.shape,i)
    for m in range(samples*(i-1),(samples*i)):
        A[k,:] = noisy_signal[m+np.arange(30)].T
        k=k+1
    #print(A.shape)
    impulse_response = np.linalg.inv(A.T*A)*A.T*clean_signal[samples*(i-1):samples*i]
    filtered_signal = sp.lfilter(np.array(np.flipud(impulse_response).T)[0],[1],np.array(noisy_signal[samples*(i-1)+1:samples*i+1].T)[0])
    array.append(filtered_signal)
    
    
r = np.concatenate(array)
result = tuple(r)
display(ipd.Audio(result, rate = fs ))
print(len(r))
rl = list(r)

#plot
w,Hinput=sp.freqz(noisy_signal[:22400]);
w,Hfsiganl=sp.freqz(result[:22400]);
w,Hspeech=sp.freqz(clean_signal[:22400]);

plt.figure(figsize=(8,6))
plt.plot(w,20*np.log10(abs(Hinput)))
plt.plot(w,20*np.log10(abs(Hfsiganl)))
plt.plot(w,20*np.log10(abs(Hspeech))); 


plt.xlabel('Normalized Frequency')
plt.ylabel('Magnitude (dB)')
plt.legend(('noisy input','filterd signal','clean signal'))
plt.title('Magnitude Spectrum')
plt.grid()

#MSE
MSE = (np.sum(np.power(noisy_signal[:22400]-clean_signal[:22400].T,2))/22400)

FMSE = (np.sum(np.power(result-clean_signal[:22400].T,2))/22400)        
type(FMSE)
print(MSE)
print(FMSE)


#SNR of the noisy signal
n = np.subtract((clean_signal[:22400]),(noisy_signal[:22400]))
#N = list(n)
noise_signal = (np.sum(np.power(noisy_signal[:22400],2)))*(1/22400)
noise = (np.sum(np.power(n[:22400],2)))*(1/22400)
#print("noise signal:",noise_signal)
snr = noise_signal/noise
print("SNR of noisy signal:",snr)
snr_before_db = 10 * np.log10(snr)
print("SNR of noisy signal in db:",snr_before_db)

#snr of the filtered sinal
#residual_noise = np.subtract((clean_signal[:22400]),(r)) 
#print(residual_noise)
filtered_signal = (np.sum(np.power(result[:22400],2)))/22400
r_noise =(np.sum(np.power(n[:22400],2)))/22400
#print("filtered signal:",filtered_signal)
snr1 = filtered_signal/r_noise
print("SNR of filtered signal:",snr1)
snr_before_db1 = 10 * np.log10(snr1)
print("SNR of filtered signal in db:",snr_before_db1)
