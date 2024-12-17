import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

x = [0, 0, 1, 1, 1, 1, 1, 1]
h = [1, 1, 2]

n_x = np.arange(len(x))
n_h = np.arange(len(h))
n_y = np.arange(len(x) + len(h) - 1)
y = np.convolve(x, h)

fig, ax = plt.subplots()
ax.set_xlim(-1, len(n_y))
ax.set_ylim(min(min(x), min(h), min(y)) - 1, max(max(x), max(h), max(y)) + 1)
ax.set_title("Discrete-Time Convolution Animation")
ax.set_xlabel("n")
ax.set_ylabel("Amplitude")

line_x, _, _ = ax.stem(n_x, x, linefmt="C0-", markerfmt="C0o", basefmt="C0-", label="x[n]")
line_h, _, _ = ax.stem(n_y, np.zeros_like(n_y), linefmt="C1-", markerfmt="C1o", basefmt="C1-", label="h[k]")
line_y, _, _ = ax.stem(n_y, np.zeros_like(n_y), linefmt="C2-", markerfmt="C2o", basefmt="C2-", label="y[n]")
ax.legend()


def update(frame):
    k = frame
    h_shifted = np.zeros(len(n_y))
    start = k
    end = start + len(h)
    
    if end <= len(n_y):
        h_shifted[start:end] = h[::-1]
    else:  
        overlap_len = len(n_y) - start
        h_shifted[start:] = h[:overlap_len][::-1]
    line_h.set_data(n_y, h_shifted)
    y_display = y[:k+1] 
    line_y.set_data(n_y[:len(y_display)], y_display)

    return line_h, line_y

frames = len(y)
ani = FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)

ani.save("discrete_convolution_animation.gif", writer=PillowWriter(fps=1))

plt.show()
