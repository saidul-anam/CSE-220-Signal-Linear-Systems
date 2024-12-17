import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def x(t):
    return np.exp(-t) * (t >= 0)

def h(t):
    return np.where((t >= 0), 1, 0)

t_min, t_max = -5, 10
dt = 0.02
t = np.arange(t_min, t_max, dt)

shifted_h_all = np.array([h(-(t - tau)) for tau in t])
y = np.convolve(x(t), h(t), mode='full') * dt
t_y = np.arange(2 * t_min, 2 * t_max - dt, dt)


fig, ax = plt.subplots()
ax.set_xlim(t_min, t_max)
ax.set_ylim(-0.1, 2.0)
ax.set_title("Continuous-Time Convolution Animation")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Amplitude")
line_x, = ax.plot([], [], 'b-', label="x(t)")
line_h, = ax.plot([], [], 'g-', label="h(t-shift)")
line_y, = ax.plot([], [], 'r-', label="y(t)")
ax.legend()

def update(frame):
    line_x.set_data(t, x(t))
    line_h.set_data(t, shifted_h_all[frame])
    line_y.set_data(t_y[:frame], y[:frame])
    return line_x, line_h, line_y

ani = FuncAnimation(fig, update, frames=len(t), interval=30, blit=True)

writer = PillowWriter(fps=15)
ani.save("continuous_convolution.gif", writer=writer)

plt.show()
