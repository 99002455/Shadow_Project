from scipy import signal as sp
from scipy import stats
import numpy as np
import IPython.display as ipd
import librosa as lbr
import wave
import struct
import matplotlib.pyplot as plt
import IPython.display as ipd

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

#calculate fft of the noisy signal
nsfft = np.fft.fft(noisy_signal)
#calculate th noise
noise = np.subtract(noisy_signal,clean_signal)
#calculate the PSD of the original signal using welch function
f, Pxx_den = sp.welch(clean_signal, window='hamming', noverlap=50)
#calculate the PSD of the noise using welch function
n, Pxx_den1 = sp.welch(noise, window='hamming', noverlap=50)
#calculate impulse response
h = Pxx_den/(Pxx_den+Pxx_den1)
#initialise result
result=[]
#initialise fft length
fft_len = 129

#multiplication of noisy signal with impulse response
iteration = int(22529/fft_len)
for i in range(1,iteration+1):
  result[fft_len*(i-1):fft_len*i] = nsfft[fft_len*(i-1):fft_len*i]*h

r = np.fft.ifft(result)
display(ipd.Audio(r, rate = fs ))
print(len(r))


#SNR of the noisy signal
noise_signal = (np.sum(np.power(noisy_signal,2)))/len(noisy_signal)
n = (np.sum(np.power(noise,2)))/len(noise)
snr = noise_signal/n
print("SNR of Noisy Signal:",snr)
snr_before_db = 10 * np.log10(snr)
print("SNR of Noisy Signal in db:",snr_before_db)

#snr of the filtered signal
filtered_signal = (np.sum(np.power(r,2)))/len(r)
r_noise =(np.sum(np.power(noise,2)))/len(noise)
snr1 = filtered_signal/r_noise
print("SNR of Filtered Signal:",abs(snr1))
snr_after_db1 = 10 * np.log10(snr1)
print("SNR of Filtered Signal in db:",abs(snr_after_db1))

