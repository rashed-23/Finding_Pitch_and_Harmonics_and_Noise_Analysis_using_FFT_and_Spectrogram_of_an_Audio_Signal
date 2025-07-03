######## Design an algorithm to identify the pitch, harmonics and noise in an audio signal.(Using FFT and Specctrogram)

''''
Sample Output: 
Pitch: 48.25Hz
Harmonics:  [96.5, 144.75, 193.0, 241.25, 289.5, 337.75, 386.0, 434.25, 482.5, 530.75, 579.0, 627.25, 675.5, 723.75, 772.0, 820.25, 868.5, 916.75]
'''''
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy.signal.windows as window

######### FAST FOURIAR TRANSFORM CALCULATION WITH THE GENERAL EQUATION  ##########
def MY_FFT(x):
    N = len(x)
    if N <= 1:
        return x
    # Divide the array into even and odd indexed elements
    even = MY_FFT(x[0::2])
    odd = MY_FFT(x[1::2])
    ### Twiddle factor calculation
    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    ### Combine the results
    return [even[k] + T[k] for k in range(N // 2)] + \
        [even[k] - T[k] for k in range(N // 2)]


########## INVERSE FAST FOURIAR TRANSFORM CALCULATION WITH THE GENERAL EQUATION  ##########
def MY_IFFT(x):
    N = len(x)
    if N <= 1:
        return x
    # Divide the array into even and odd indexed elements
    even = MY_IFFT(x[0::2])
    odd = MY_IFFT(x[1::2])

    # Combine the results
    T = [np.exp(2j * np.pi * k / N) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + \
        [even[k] - T[k] for k in range(N // 2)]

def MY_FFT_Frequencies(N,sr):
    y = np.arange(0, N// 2) * (sr / N)
    return y

def MY_Zero_Padding(y):
    l=len(y)
    if l&(l-1):
        y = np.pad(y, (0,np.power(2,int(np.log2(l))+1)-l))
    return y

########################################## Main Code###################################

audio_path = r"D:\sample-3s.mp3" ## USE any audio file path on your computer.##

main_signal, sr = librosa.load(audio_path,sr=4096,mono=True) # y is the audio time series, sr is the sampling rate


###################################### FFT CALCULATION #################################

l=len(main_signal)# Number of samples
main_signal=MY_Zero_Padding(main_signal) # Zero Padding
FFT_signal = MY_FFT(main_signal)   # FFT Calculation
N=len(main_signal)
main_signal=main_signal[:l]
main_signal/=max(main_signal) ## Normalise
FFT_signal=np.abs(FFT_signal)  #Taking the magnitude of the frequencies
FFT_signal/=max(FFT_signal) ## Normalise
frequencies=MY_FFT_Frequencies(N,sr) # Frequency Calculation by f=k*fs/N
FFT_half=FFT_signal[:N//2]  # Taking only positive frequencies

######################################### Pitch Calculation ###########################

pitch_ind=np.argmax(FFT_half[1:])+1 #Max Pick index
pitch =frequencies[pitch_ind]
print(f"Pitch: {pitch}Hz")

################################## Harmonics Calculation ##############################

harmonic_ind=[]
harmonics=[]
for i in range(2,20):
    ind=np.argmin(np.abs(frequencies-i*pitch))
    harmonic_ind.append(ind)
    harmonics.append(frequencies[ind])
print("Harmonics: ",harmonics)

################################## Ploting #######################################

#Main signal
t = np.arange(l) / sr
plt.subplot(2, 2, 1)
plt.stem(t,main_signal)
plt.title('Original Audio Signal',fontweight='bold')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
#plt.show()

# FFT signal
plt.subplot(2, 2, 2)
plt.stem(frequencies,FFT_half)
plt.title('FFT Magnitude Spectrum',fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
#plt.show()

# FFT signal with Pitch, Harmonics
plt.subplot(2, 2, 3)
plt.stem(frequencies,FFT_half)
plt.title('FFT Magnitude Spectrum with Pitch and Harmonics',fontweight='bold')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.stem(frequencies[pitch_ind],FFT_half[pitch_ind], 'r', label='Fundamental Frequency')
plt.stem(frequencies[harmonic_ind], FFT_half[harmonic_ind], 'y', label='Harmonics')
plt.legend()
plt.grid()
#plt.show()

# Spectrogram
plt.subplot(2, 2, 4)
plt.specgram(main_signal, Fs=sr, NFFT=2048, noverlap=1024,window=window.hamming(2048), cmap='viridis')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title('Spectrogram',fontweight='bold')
plt.subplots_adjust(hspace=.34)
plt.savefig('audio_analysis.png', dpi=300)
plt.show()




