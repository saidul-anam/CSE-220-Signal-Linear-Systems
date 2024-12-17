import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import math


class ContinuousSignal:
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        self.func = func
    
    def shift(self, shift: float):
        return ContinuousSignal(lambda t: self.func(t - shift))
    
    def add(self, other):
        return ContinuousSignal(lambda t: self.func(t) + other.func(t))
    
    def multiply(self, other):
        return ContinuousSignal(lambda t: self.func(t) * other.func(t))
    
    def multiply_const_factor(self, scalar: float):
        return ContinuousSignal(lambda t: self.func(t) * scalar)
    
    def plot(self,savepath:None, t_range=(-10, 10), num_points=1000, title='Continuous Signal', x_label='t', y_label='x(t)'):
        t = np.linspace(t_range[0], t_range[1], num_points)
        signal_values = self.func(t)
        
        plt.figure(figsize=(8, 4))
        plt.plot(t, signal_values)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        plt.savefig(savepath)



class LTIContinuous:
   def __init__(self, impulse_response: ContinuousSignal):
        self.impulse_response = impulse_response
    
   def linear_combination_of_impulses(self, input_signal: ContinuousSignal, delta: float):
        t = np.arange(-3, 3+delta, delta)
        signal_values = input_signal.func(t)
        impulses = []
        coefficients = []
        for i, val in enumerate(signal_values):
                impulses.append(t[i])
                coefficients.append(val * delta)
        return impulses, coefficients       
    
   def output_approx(self, input_signal: ContinuousSignal, delta: float):
        t = np.arange(-3, 3+delta,delta)
        output_signal=input_signal.func(t)*delta
        return t, output_signal
   
   # delta function 

   def delt(self,t,d):
        return np.where((t >= 0) & (t < d), 1/d, 0)
   
   # impulse signal

   def impulse_sig(self,t):
        return np.where((t >= 0), 1, 0)
   
   # 
   def subplotInput(self, input_signal, delta):
    impulses, coefficients = self.linear_combination_of_impulses(input_signal, delta)
    fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10, 10))
    axes = axes.flatten()
    t = np.linspace(-3, 3,1000)
    z=np.zeros_like(t)
    for idx, ax in enumerate(axes[:-1]):
            if idx >= len(coefficients):
              break
            print(idx)
            time=delta*idx-3
            input=self.delt(t-time,delta)
            input=coefficients[idx]*input
            print(coefficients[idx])
            z+=input
            ax.plot(t, input)
            ax.set_title(r'$\delta(t - ({0}))x({0})$'.format(round(impulses[idx], 2), round(coefficients[idx], 2)))
            ax.set_xlabel('t (Time)')
            ax.set_ylabel('x(t)')
            ax.set_xlim(-3,3)
            ax.set_ylim(-1,3)
            
    axes[-1].plot(t,z)
    axes[-1].set_title('Reconstructed Signal')
    axes[-1].set_xlabel('t (Time)')
    axes[-1].set_ylabel('x(t)')
    axes[-1].set_xlim(-3,3)
    axes[-1].set_ylim(-1,3)
    plt.tight_layout()
    plt.savefig(f'{'.'}/ Returned impulses multiplied by their coefficients.png')
   


   def funcVarDel(self,input_signal,delta:float):
    impulses, coefficients = self.linear_combination_of_impulses(input_signal, delta)
    t = np.linspace(-3, 3,1000)
    z=np.zeros_like(t)
    for i in range(len(impulses)):
            time=delta*i-3
            input1=self.delt(t-time,delta)
            input1=coefficients[i]*input1
            z+=input1   
    return z   
   def subplotVarDel(self, input_signal):
    t = np.linspace(-3, 3,1000)
    input=input_signal.func(t)
    z1=self.funcVarDel(input_signal,0.5)
    z2=self.funcVarDel(input_signal,0.1)
    z3=self.funcVarDel(input_signal,0.05)
    z4=self.funcVarDel(input_signal,0.01)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,5))
    axes[0, 0].plot(t, z1)
    axes[0, 0].plot(t, input)
    axes[0, 0].set_xlim(-3, 3)
    axes[0, 0].set_ylim(-1, 3)
    axes[0, 0].set_title("delta=0.5")
    axes[0, 0].set_xlabel('t (Time)')
    axes[0, 0].set_ylabel('x(t)')
    
    axes[0, 1].plot(t, z2)
    axes[0, 1].plot(t, input)
    axes[0, 1].set_xlim(-3, 3)
    axes[0, 1].set_ylim(-1, 3)
    axes[0, 1].set_title("delta=0.1")
    axes[0, 1].set_xlabel('t (Time)')
    axes[0, 1].set_ylabel('x(t)')
    
    axes[1, 0].plot(t, z3)
    axes[1, 0].plot(t, input)
    axes[1, 0].set_xlim(-3, 3)
    axes[1, 0].set_ylim(-1, 3)
    axes[1, 0].set_title("delta=0.05")
    axes[1, 0].set_xlabel('t (Time)')
    axes[1, 0].set_ylabel('x(t)')
    
    axes[1, 1].plot(t, z4)
    axes[1, 1].plot(t, input)
    axes[1, 1].set_xlim(-3, 3)
    axes[1, 1].set_ylim(-1, 3)
    axes[1, 1].set_title("delta=0.01")
    axes[1, 1].set_xlabel('t (Time)')
    axes[1, 1].set_ylabel('x(t)')
    plt.tight_layout()
    plt.savefig(f'{'.'}/ Reconstruction of input signal with varying delta.png')

   def showsubh(self, input_signal, delta: float):
    fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10, 10))
    axes = axes.flatten()
    time, output = self.output_approx(input_signal, delta)
    t = np.linspace(-3, 3 ,1000)
    z = np.zeros_like(t)  
    for idx, ax in enumerate(axes[:-1]):
        if idx >= len(output):
            break
        tim = delta * idx - 3
        input_impulse = np.zeros_like(t)
        input_impulse = output[idx]*self.impulse_sig(t-tim)
        z += input_impulse
        ax.plot(t, input_impulse)
        ax.set_title(r'$h(t - ({0})) \times x({0})$'.format(round(tim, 2)))
        ax.set_xlabel('t (Time)')
        ax.set_ylabel('x(t)')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 1)
    axes[-1].plot(t, z)
    axes[-1].set_title('Reconstructed Signal')
    axes[-1].set_xlabel('t (Time)')
    axes[-1].set_ylabel('x(t)')
    axes[-1].set_xlim(-3, 3)
    axes[-1].set_ylim(-1, 2)
    plt.tight_layout()
    plt.savefig('./Reconstruction_of_input_signal_with_varying_delta.png')

   def funcVarDelout(self,input_signal,delta:float):
    time, output = self.output_approx(input_signal, delta)
    t = np.linspace(-3, 3,1000)
    z=np.zeros_like(t)
    for i in range(len(output)):
        tim = delta * i - 3
        input_impulse = np.zeros_like(t)
        input_impulse = output[i]*self.impulse_sig(t-tim)
        z += input_impulse  
    return z  
   def subplotVarDelout(self, input_signal):
    t = np.linspace(-3, 3,1000)
    input=self.funcVarDelout(input_signal,0.001)
    z1=self.funcVarDelout(input_signal,0.5)
    z2=self.funcVarDelout(input_signal,0.1)
    z3=self.funcVarDelout(input_signal,0.05)
    z4=self.funcVarDelout(input_signal,0.01)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,5))
    axes[0, 0].plot(t, z1)
    axes[0, 0].plot(t, input)
    axes[0, 0].set_xlim(-3, 3)
    axes[0, 0].set_ylim(-1, 1.5)
    axes[0, 0].set_title("delta=0.5")
    axes[0, 0].set_xlabel('t (Time)')
    axes[0, 0].set_ylabel('x(t)')
    
    axes[0, 1].plot(t, z2)
    axes[0, 1].plot(t, input)
    axes[0, 1].set_xlim(-3, 3)
    axes[0, 1].set_ylim(-1, 1.5)
    axes[0, 1].set_title("delta=0.1")
    axes[0, 1].set_xlabel('t (Time)')
    axes[0, 1].set_ylabel('x(t)')
    
    axes[1, 0].plot(t, z3)
    axes[1, 0].plot(t, input)
    axes[1, 0].set_xlim(-3, 3)
    axes[1, 0].set_ylim(-1, 1.5)
    axes[1, 0].set_title("delta=0.05")
    axes[1, 0].set_xlabel('t (Time)')
    axes[1, 0].set_ylabel('x(t)')
    
    axes[1, 1].plot(t, z4)
    axes[1, 1].plot(t, input)
    axes[1, 1].set_xlim(-3, 3)
    axes[1, 1].set_ylim(-1, 1.5)
    axes[1, 1].set_title("delta=0.01")
    axes[1, 1].set_xlabel('t (Time)')
    axes[1, 1].set_ylabel('x(t)')
    plt.tight_layout()
    plt.savefig(f'{'.'}/  Approximate output with varying delta.png')  





class DiscreteSignal:
    def __init__(self, INF: int):
        self.INF = INF
        self.values = np.zeros(2 * INF + 1)
    
    def set_value_at_time(self, time: int, value: float):
        index = time + self.INF
        self.values[index] = value
    
    def shift_signal(self, shift: int):
        shifted_signal = DiscreteSignal(self.INF)
        shifted_signal.values = np.roll(self.values, shift)
        return shifted_signal
    
    def add(self, other):
        if self.INF != other.INF:
            raise ValueError("signals must have same INF")
        result_signal = DiscreteSignal(self.INF)
        result_signal.values = self.values + other.values
        return result_signal
    
    def multiply(self, other):
        if self.INF != other.INF:
            raise ValueError("signals must have same INF")
        
        result_signal = DiscreteSignal(self.INF)
        result_signal.values = self.values * other.values
        return result_signal
    
    def multiply_const_factor(self, scalar: float):
        result_signal = DiscreteSignal(self.INF)
        result_signal.values = self.values * scalar
        return result_signal
    
    def plot(self,savepath:None, title: str = 'Discrete Signal', y_range=(-1, 3)):
        y_range = (min(y_range[0],np.min(self.values)), max(np.max(self.values), y_range[1]) + 1)
        time_indices = np.arange(-self.INF, self.INF + 1)
        plt.figure(figsize=(8, 3))
        plt.stem(time_indices, self.values)
        plt.title(title)
        plt.xlabel('n (Time Index)')
        plt.ylabel('x[n]')
        plt.ylim(y_range)
        plt.grid(True)
        plt.savefig(savepath)

class LTIDiscrete:
    def __init__(self, impulse_response: DiscreteSignal):
        self.impulse_response = impulse_response
    
    def linear_combination_of_impulses(self, input_signal: DiscreteSignal):
        impulses = []
        coefficients = []
        for n in range(-input_signal.INF, input_signal.INF + 1):
            value = input_signal.values[n + input_signal.INF]
            if value!=0:
               impulses.append(n)
               coefficients.append(value)   
        return impulses, coefficients        
    
    def output(self, input_signal: DiscreteSignal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        output_values = np.zeros(2 * input_signal.INF + 1)
        for impulse_time, coefficient in zip(impulses, coefficients):
            shifted_response = self.impulse_response.shift_signal(impulse_time).values
            output_values += coefficient * shifted_response
        
        return output_values

        

    def subplotOut(self,input_signal: DiscreteSignal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        outputvalues=self.output(input_signal)
        y_range=(-1,3)
        y_range = (min(y_range[0],np.min(outputvalues)), max(np.max(outputvalues), y_range[1]) + 2)
        time_indices = np.arange(-input_signal.INF, input_signal.INF + 1)
        row=int((2*(input_signal.INF)+2)/3)
        row=math.ceil(row)
        fig, axes = plt.subplots(row, 3, sharex=True, sharey=True,figsize=(10, 5))
        axes = axes.flatten()
        y=0
        for x in range(2*input_signal.INF+1):
            output=np.zeros(2 * input_signal.INF + 1)
            if y < len(impulses) and impulses[y]==x-input_signal.INF:
                output=coefficients[y]*self.impulse_response.shift_signal(impulses[y]).values
                y=y+1
            axes[x].stem(time_indices, output)
            axes[x].set_xlabel('n(Time Index)')
            axes[x].set_ylabel('x[n]')
            axes[x].set_title(f'h[n-({x - input_signal.INF})] * x[{x - input_signal.INF}]')
            axes[x].set_ylim(y_range)

        axes[-1].stem(time_indices, outputvalues)
        axes[-1].set_xlabel('n(Time Index)')
        axes[-1].set_ylabel('x[n]')
        axes[-1].set_title('Output Sum')
        axes[-1].set_ylim(y_range)
        plt.tight_layout()
        plt.savefig(f'{'.'}/Output.png')


    def subplotInput(self,input_signal: DiscreteSignal):
        impulses, coefficients = self.linear_combination_of_impulses(input_signal)
        y_range=(-1,3)
        y_range = (min(y_range[0],np.min(input_signal.values)), max(np.max(input_signal.values), y_range[1]) + 2)
        time_indices = np.arange(-input_signal.INF, input_signal.INF + 1)
        row=int((2*(input_signal.INF)+2)/3)
        row=math.ceil(row)
        fig, axes = plt.subplots(row, 3, sharex=True, sharey=True,figsize=(10, 5))
        axes = axes.flatten()
        y=0
        for x in range(2*input_signal.INF+1):
            input=np.zeros(2 * input_signal.INF + 1)
            if y < len(impulses) and impulses[y]==x-input_signal.INF:
                  input[x]=coefficients[y]
                  y=y+1
            axes[x].stem(time_indices, input)
            axes[x].set_xlabel('n(Time Index)')
            axes[x].set_ylabel('x[n]')
            axes[x].set_title(r'$\delta$'f'[n-({x - input_signal.INF})] * x[{x - input_signal.INF}]')
            axes[x].set_ylim(y_range)

        axes[-1].stem(time_indices, input_signal.values)
        axes[-1].set_xlabel('n(Time Index)')
        axes[-1].set_ylabel('x[n]')
        axes[-1].set_title('Output Sum')
        axes[-1].set_ylim(y_range)
        plt.tight_layout()
        plt.savefig(f'{'.'}/Returned impulses multiplied by respective coefficients.png')
        
        
def main():
    signal1 = DiscreteSignal(INF=5)
    signal1.set_value_at_time(0, .5)
    signal1.set_value_at_time(1, 2)
    signal1.plot(title=" Input Discrete Signal, INF = 5",savepath="./input_discrete_signal.png")
    signal2 = DiscreteSignal(INF=5)
    impulse_response = DiscreteSignal(INF=5)
    impulse_response.set_value_at_time(0, 1)
    impulse_response.set_value_at_time(1, 1)
    impulse_response.set_value_at_time(-1, 1) 
    impulse_response.plot(title='Impulse Response h[n]',savepath="./Impulse Responseh[n].png")
    lti_system = LTIDiscrete(impulse_response)
    lti_system.subplotInput(signal1)
    lti_system.subplotOut(signal1)
    

main()
def Cmain():
    continuous_function = lambda t:(np.exp(-t)) * (t >= 0)
    delta = 0.5
    continuous_function1 = lambda t:(1) * (t >= 0)
    signal = ContinuousSignal(continuous_function)
    signal.plot(title='Continuous Function f(t)',savepath='./Continuous Function f(t).png')
    impulse_response = ContinuousSignal(continuous_function1)
    lti_system = LTIContinuous(impulse_response)
    lti_system.subplotInput(signal,delta)
    lti_system.subplotVarDel(signal)
    lti_system.showsubh(signal,delta)
    lti_system.subplotVarDelout(signal)
    
Cmain()

