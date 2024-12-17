import numpy as np
import matplotlib.pyplot as plt
n=50
samples = np.arange(n) 
sampling_rate=100
wave_velocity=8000
#use this function to generate signal_A and signal_B with a random shift
def generate_signals(frequency=5):

    noise_freqs = [15, 30, 45]  # Default noise frequencies in Hz

    amplitudes = [0.5, 0.3, 0.1]  # Default noise amplitudes
    noise_freqs2 = [10, 20, 40] 
    amplitudes2 = [0.3, 0.2, 0.1]
    
     # Discrete sample indices
    dt = 1 / sampling_rate  # Sampling interval in seconds
    time = samples * dt  # Time points corresponding to each sample

    # Original clean signal (sinusoidal)
    original_signal = np.sin(2 * np.pi * frequency * time)

    # Adding noise
    noise_for_sigal_A = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs, amplitudes))
    noise_for_sigal_B = sum(amplitude * np.sin(2 * np.pi * noise_freq * time)
                for noise_freq, amplitude in zip(noise_freqs2, amplitudes2))
    signal_A = original_signal + noise_for_sigal_A 
    noisy_signal_B = signal_A + noise_for_sigal_B

    # Applying random shift
    shift_samples = np.random.randint(-n // 2, n // 2)  # Random shift
    print(f"Shift Samples: {shift_samples}")
    signal_B = np.roll(noisy_signal_B, shift_samples)
    
    return signal_A, signal_B

#implement other functions and logic
def compute_dft(signal):
    l=len(signal)
    sig=np.zeros(l,dtype=complex)
    for k in range(l):
        for n in range(l):
            sig[k]+=signal[n]*np.exp(-2j*np.pi*n*k/l)
    return sig


def compute_idft(signal):
    l=len(signal)
    sig=np.zeros(l,dtype=complex)
    for k in range(l):
        for n in range(l):
            sig[k]+=signal[n]*np.exp(2j*np.pi*n*k/l)
    return sig/l


def compute_cross_correlation(signalA,signalB):
   dfta=compute_dft(signalA)
   dftb=compute_dft(signalB)
   cross_cor=compute_idft(dftb*np.conjugate(dfta))
   return np.real(cross_cor)



def sample_lag(cross_cor):
    lag=np.argmax(cross_cor)
    if(lag>n//2):
       lag=lag-n
    return lag


def distance_calc(lag_samples, sampling_rate, velocity):
    lag_samples=abs(lag_samples)
    return lag_samples * (1/sampling_rate)*velocity
def plot_signal_a(signal_a):
    plt.figure(figsize=(10,5))
    plt.stem(signal_a,linefmt="b-",markerfmt="bo",basefmt=" ",label='Signal A')
    plt.xlabel('Sample (n)')
    plt.ylabel('Amplitude')
    plt.title('Signal A')
    plt.legend()
    plt.grid()
    plt.show()


def plot_signal_b(signal_a):
    plt.figure(figsize=(10,5))
    plt.stem(signal_a,linefmt="r-",markerfmt="ro",basefmt=" ",label='Signal B')
    plt.xlabel('Sample (n)')
    plt.ylabel('Amplitude')
    plt.title('Signal B')
    plt.legend()
    plt.grid()
    plt.show()

def plot_frequency_spectrum(signal,label):
    dft_result=compute_dft(signal)
    magnitude=np.abs(dft_result)
    plt.figure(figsize=(10,5))
    plt.stem(magnitude,linefmt="b-",markerfmt="bo",basefmt=" ")
    plt.xlabel('Frequency Index (k)')
    plt.ylabel('Magnitude')
    plt.title(f'Frequency Spectrum of {label}')
    plt.grid()
    plt.show()

def plot_cross_correlation(cross_corr):
    lags=np.arange(-len(cross_corr)//2, len(cross_corr)//2)
    shifted_corr=np.roll(cross_corr,len(cross_corr)//2)
    plt.figure(figsize=(10,5))
    plt.stem(lags,shifted_corr,linefmt="g-",markerfmt="go",basefmt=" ")
    plt.xlabel('Lag (samples)')
    plt.ylabel('Correlation')
    plt.title('Cross-Correlation')
    plt.grid()
    plt.show()

signala,signalb=generate_signals()
plot_signal_a(signala)
plot_signal_b(signalb)
plot_frequency_spectrum(signala,label="signal A")
plot_frequency_spectrum(signalb,label="signal B")
cross_cor=compute_cross_correlation(signala,signalb)
plot_cross_correlation(cross_cor)
lag=sample_lag(cross_cor)
print(f"Sample Lag: {lag}")

distance =distance_calc(lag,sampling_rate,wave_velocity)
print(f"Estimated Distance: {distance:.2f} meters")



