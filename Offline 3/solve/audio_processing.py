import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load the image
image = Image.open('im.jpeg').convert('L')  # Convert to grayscale
image_data = np.array(image)

# Step 1.1: Visualize the original image
plt.figure(figsize=(8, 8))
plt.imshow(image_data, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# Step 2: Apply Fourier Transform
ft_image = np.fft.fft2(image_data)
ft_image_shifted = np.fft.fftshift(ft_image)  # Shift zero frequency to the center
magnitude_spectrum = np.log(np.abs(ft_image_shifted) + 1)  # Log scale for better visualization

# Step 2.1: Visualize the magnitude spectrum
plt.figure(figsize=(8, 8))
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')
plt.show()

# Step 3: Filter out high-frequency noise
rows, cols = image_data.shape
crow, ccol = rows // 2, cols // 2  # Center of the frequency domain

# Create a mask with a low-pass filter
mask = np.zeros((rows, cols), np.uint8)
radius = 50  # Radius of the low-pass filter
mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1

# Apply the mask to the shifted Fourier transform
filtered_ft_image_shifted = ft_image_shifted * mask

# Step 3.1: Visualize the filtered magnitude spectrum
filtered_magnitude_spectrum = np.log(np.abs(filtered_ft_image_shifted) + 1)
plt.figure(figsize=(8, 8))
plt.imshow(filtered_magnitude_spectrum, cmap='gray')
plt.title("Filtered Magnitude Spectrum")
plt.axis('off')
plt.show()

# Step 4: Apply Inverse Fourier Transform to reconstruct the image
filtered_ft_image = np.fft.ifftshift(filtered_ft_image_shifted)  # Shift back
filtered_image = np.fft.ifft2(filtered_ft_image).real  # Inverse FFT

# Step 4.1: Visualize the reconstructed (filtered) image
plt.figure(figsize=(8, 8))
plt.imshow(filtered_image, cmap='gray')
plt.title("Filtered Image")
plt.axis('off')
plt.show()

print("Image filtration complete.")
