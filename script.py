import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Task 5: Synthesize Signal from Spectrum
# -------------------------------------------------------------------

# 1. Define the input spectrum vector x_mu for Task 5 (Eq. 21)
x_mu_vec = np.array([6, 4, 4, 5, 3, 4, 5, 0, 0, 0, 0])

# Determine the block length N
N = len(x_mu_vec)

print(f"--- Task 5: Signal Synthesis ---")
print(f"Block length N = {N}")
print(f"Input Spectrum x_mu = {x_mu_vec}\n")

# 2. Create the (k * mu) outer product matrix K (Eq. 9)
k_mu_range = np.arange(N)
K = np.outer(k_mu_range, k_mu_range)

print(f"--- Matrix K (N={N}) ---")
print(K)
print("\n")

# 3. Create the Fourier Matrix W (Eq. 7)
# W = exp(+j * 2*pi/N * K)
W = np.exp(1j * (2 * np.pi / N) * K)

# Print W (rounded for readability, as in the N=4 example)
print(f"--- Fourier Matrix W (N={N}) ---")
print(np.round(W, 2))
print("\n")

# 4. Synthesize the time-domain signal xk using IDFT (Eq. 6 or 13)
# xk = (1/N) * W * x_mu
# np.dot handles the matrix-vector multiplication
xk = (1 / N) * np.dot(W, x_mu_vec)

print(f"--- Synthesized Signal xk (first 5 samples) ---")
print(np.round(xk[:5], 4))
print("\n")

# 5. Verification (Optional, but good practice)
# Compare our matrix method with numpy's built-in ifft
xk_check = np.fft.ifft(x_mu_vec)
print(f"--- Verification vs. np.fft.ifft() ---")
print(f"np.fft.ifft (first 5 samples): {np.round(xk_check[:5], 4)}")
print(f"Signals match: {np.allclose(xk, xk_check)}")
print("\n")


# 6. Plot the synthesized signal xk
# The signal is complex, so we plot its real and imaginary parts
k_axis = np.arange(N)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Plot Real Part
ax1.stem(k_axis, np.real(xk), basefmt="k-")
ax1.set_title(f'Synthesized Signal x[k] for N={N} (Task 5)')
ax1.set_ylabel('Amplitude (Real Part)')
ax1.grid(True)

# Plot Imaginary Part
ax2.stem(k_axis, np.imag(xk), 'r', markerfmt='ro', basefmt="k-")
ax2.set_ylabel('Amplitude (Imaginary Part)')
ax2.set_xlabel('Sample Index k')
ax2.set_xticks(k_axis)  # Ensure all discrete k values are shown
ax2.grid(True)

plt.tight_layout()
plt.show()