import numpy as np
import matplotlib.pyplot as plt

# prior distribution
s = np.linspace(-np.pi, np.pi, 100)
p_s = 2 - np.abs(np.sin(s))
p_s = p_s / np.sum(p_s)  # normalise

# create function that warps orientation using the
cdf_p_s = np.cumsum(p_s)

F = lambda x: np.interp(x, s, cdf_p_s * 2*np.pi - np.pi)
Finv = lambda x: np.interp(x, cdf_p_s * 2*np.pi - np.pi, s)

# circular tuning curves for M neurons
M = 8
phis = np.arange(-np.pi, np.pi, 2*np.pi/M)
kappa = 3
tf = lambda x: np.exp(kappa*np.cos(x) - kappa)

tuning = tf(s[:, None] - phis[None, :])
warped_tuning = tf(F(s)[:, None] - phis[None, :])

plt.figure(figsize=(12, 3))

plt.subplot(1, 4, 1)
plt.plot(s, p_s)
plt.xlabel("Orientation")
plt.ylabel("Probability")
plt.title("Orientation prior")

plt.subplot(1, 4, 2)
plt.plot(s, cdf_p_s)
plt.xlabel("Orientation")
plt.ylabel("Cumulative probability")
plt.title("Cumulative prior")

plt.subplot(1, 4, 3)
plt.plot(s, tuning)
plt.xlabel("Orientation")
plt.ylabel("Firing rate")
plt.title("Tuning curves")

plt.subplot(1, 4, 4)
plt.plot(s, warped_tuning)
plt.xlabel("Orientation")
plt.ylabel("Firing rate")
plt.title("Warped tuning curves")

plt.tight_layout()
plt.show()