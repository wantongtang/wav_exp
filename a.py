import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

from scipy import signal
from scipy.io import wavfile
from pydub import AudioSegment

train_audio_path = './'
filename = 'dd.wav'
sample_rate, samples = wavfile.read(str(train_audio_path) + filename)
#print np.shape(samples)

#sound = AudioSegment.from_file(str(train_audio_path) + filename)
#sample_rate = sound.frame_rate
#samples = np.array(sound.get_array_of_samples()).reshape(sample_rate,-1)
#print np.shape(samples)




def log_specgram(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,fs=sample_rate,window='hann',nperseg=nperseg,noverlap=noverlap,detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

freqs, times, spectrogram = log_specgram(samples, sample_rate)

fig = plt.figure(figsize=(14, 8))
#ax1 = fig.subplots()#fig.add_subplot(211)
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + filename)
ax1.set_ylabel('Amplitude')
#ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)
ax1.plot( samples)

ax2 = fig.add_subplot(212)
ax2.imshow(spectrogram.T, aspect='auto', origin='lower', 
                   extent=[times.min(), times.max(), freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[::16])
ax2.set_title('Spectrogram of ' + filename)
ax2.set_ylabel('Freqs in Hz')
ax2.set_xlabel('Seconds')
#plt.show()

plt.figure()
plt.plot(samples[8000:12000:1])
#plt.show()


plt.figure()
plt.plot(samples[0:100:1])
#plt.show()
def custom_fft(y,fs):
    T = 1.0/fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0,1.0/(2.0*T),N//2)
    vals = 2.0/N*np.abs(yf[0:N//2])
    return xf,vals
xf,vals = custom_fft(samples,sample_rate)
plt.figure(figsize=(12,4))
plt.title("fft")
plt.plot(xf,vals)
plt.xlabel("Frequency")
plt.grid()
plt.show()


