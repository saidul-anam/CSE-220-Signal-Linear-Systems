import numpy as np
import matplotlib.pyplot as plt
import time


sample_sizes=[2**k for k in range(2, 11)]
dft_times=[]
idft_times=[]
fft_times=[]
ifft_times=[]


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


def compute_fft(signal):
    n=len(signal)
    if(n==1):
        return signal
    even=compute_fft(signal[0::2])
    odd=compute_fft(signal[1::2])
    terms=[np.exp(-2j*np.pi*k/n)*odd[k] for k in range(n//2)]
    return np.concatenate([even+terms,even-terms])


def compute_ifft(signal):
    n=len(signal)
    conj=np.conjugate(signal)
    trans=compute_fft(conj)
    trans=np.conjugate(trans)
    return trans/n



def measure_runtime(func, signal):
    start=time.time()
    func(signal)
    return time.time()-start



def generate_random_signal(size):
    return np.random.rand(size)



for size in sample_sizes:
    signal=generate_random_signal(size)
    dft_time=np.mean([measure_runtime(compute_dft,signal) for _ in range(10)])
    dft_times.append(dft_time)
    idft_time=np.mean([measure_runtime(compute_idft,signal) for _ in range(10)])
    idft_times.append(idft_time)
    fft_time=np.mean([measure_runtime(compute_fft,signal) for _ in range(10)])
    fft_times.append(fft_time)
    ifft_time=np.mean([measure_runtime(compute_ifft,signal) for _ in range(10)])
    ifft_times.append(ifft_time)


    
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes,dft_times,label='DFT Runtime',marker='o')
plt.plot(sample_sizes,idft_times,label='IDFT Runtime',marker='o')
plt.plot(sample_sizes,fft_times,label='FFT Runtime',marker='o')
plt.plot(sample_sizes,ifft_times,label='IFFT Runtime',marker='o')
plt.xlabel('Input Size')
plt.ylabel('Average Runtime (s)')
plt.title('Runtime Comparison:DFT,IDFT,FFT,and IFFT')
plt.legend()
plt.grid()
plt.show()

    