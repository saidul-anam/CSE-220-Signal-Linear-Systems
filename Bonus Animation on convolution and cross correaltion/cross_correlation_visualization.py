import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def generate_signals():
    n = np.arange(50)
    x = np.sin(2 * np.pi * 0.1 * n)
    y = np.roll(x, shift=10) + 0.1 * np.random.randn(len(n))
    return x, y
def compute_cross_correlation(x, y):
    N = len(x)
    r = np.correlate(x, y, mode='full')
    lags = np.arange(-N + 1, N)
    return r, lags
def animate(i):
    global y_shifted, r_values, peak_index
    y_shifted = np.roll(y_original, shift=i)
    ax[0].cla()
    ax[0].stem(x_original, linefmt="b-", markerfmt="bo", basefmt=" ", label="x(n)")
    ax[0].stem(y_shifted, linefmt="r-", markerfmt="ro", basefmt=" ", label="y(n)")
    ax[0].set_title("Dynamic Shift of y(n)")
    ax[0].set_xlabel("Sample")
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()
    ax[0].grid()

    r_values, _ = compute_cross_correlation(x_original, y_shifted)
    line_corr.set_ydata(r_values)
    peak_index = np.argmax(r_values)
    marker_corr.set_data([lags[peak_index]], [r_values[peak_index]])

    return line_corr, marker_corr
x_original, y_original = generate_signals()
y_shifted = y_original.copy()
r_values, lags = compute_cross_correlation(x_original, y_original)
peak_index = np.argmax(r_values)
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].stem(x_original, linefmt="b-", markerfmt="bo", basefmt=" ", label="x(n)")
ax[0].stem(y_original, linefmt="r-", markerfmt="ro", basefmt=" ", label="y(n)")
ax[0].set_title("Dynamic Shift of y(n)")
ax[0].set_xlabel("Sample")
ax[0].set_ylabel("Amplitude")
ax[0].legend()
ax[0].grid()
line_corr, = ax[1].plot(lags, r_values, "g-", label="Cross-Correlation r(n)")
marker_corr, = ax[1].plot([lags[peak_index]], [r_values[peak_index]], "ro", label="Peak Correlation")
ax[1].set_title("Cross-Correlation")
ax[1].set_xlabel("Lag (samples)")
ax[1].set_ylabel("Correlation")
ax[1].legend()
ax[1].grid()

ani = FuncAnimation(
    fig,
    animate,
    frames=len(x_original),  
    interval=1000,           
    blit=False            
)

ani.save("cross_correlation_simulation.gif", writer="pillow", fps=10)  
print("Animation saved as cross_correlation_simulation.gif")

plt.show()
