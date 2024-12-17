import numpy as np
import matplotlib.pyplot as plt

def parabolic_func(x):
    return x**2
def triangular_wave(x):
    return np.maximum(1 - np.abs(x), 0)
def sawtooth_wave(x):
    return np.where((x >= -2) & (x <= 2), (x+2) / 2, 0)
def rectangular_wave(x):
    return np.where((x >= -2) & (x <= 2), 1, 0)
# Define the interval and function and generate appropriate x values and y values
x_values = np.linspace(-10,10,1000)
y_values = np.where((x_values>=-2)&(x_values<=2),parabolic_func(x_values),0) 

# Plot the original function
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2")
plt.title("Original Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# Define the sampled times and frequencies
sampled_times = x_values
frequencies = np.linspace(-2,2,200)

# Fourier Transform 
def fourier_transform(signal, frequencies, sampled_times):
    num_freqs = len(frequencies)
    ft_result_real = np.zeros(num_freqs)
    ft_result_imag = np.zeros(num_freqs)
    
    # Store the fourier transform results for each frequency. Handle the real and imaginary parts separately
    # use trapezoidal integration to calculate the real and imaginary parts of the FT
    for i,freq in enumerate(frequencies):
        real_cos=np.cos(2*np.pi*freq*sampled_times)
        imag_sin=np.sin(2*np.pi*freq*sampled_times)
        ft_result_real[i]=np.trapz(signal*real_cos,sampled_times)
        ft_result_imag[i]=-np.trapz(signal*imag_sin,sampled_times)
    return ft_result_real, ft_result_imag

# Apply FT to the sampled data
ft_data = fourier_transform(y_values, frequencies, sampled_times)
 #plot the FT data
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.sqrt(ft_data[0]**2 + ft_data[1]**2))
plt.title("Frequency Spectrum of y = x^2")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()


# Inverse Fourier Transform 
def inverse_fourier_transform(ft_signal, frequencies, sampled_times):
    n =len(sampled_times)
    reconstructed_signal = np.zeros(n)
    # Reconstruct the signal by summing over all frequencies for each time in sampled_times.
    # use trapezoidal integration to calculate the real part
    # You have to return only the real part of the reconstructed signal
    for i,t in enumerate(sampled_times):
        cos_value=np.cos(2*np.pi*frequencies*t)
        sin_value=np.sin(2*np.pi*frequencies*t)
        real_part=np.trapz(ft_signal[0]*cos_value-ft_signal[1]*sin_value,frequencies)
        reconstructed_signal[i]=real_part
    
    return reconstructed_signal

# Reconstruct the signal from the FT data
reconstructed_y_values = inverse_fourier_transform(ft_data, frequencies, sampled_times)
# Plot the original and reconstructed functions for comparison
plt.figure(figsize=(12, 4))
plt.plot(x_values, y_values, label="Original y = x^2", color="blue")
plt.plot(sampled_times, reconstructed_y_values, label="Reconstructed y = x^2", color="red", linestyle="--")
plt.title("Original vs Reconstructed Function (y = x^2)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
