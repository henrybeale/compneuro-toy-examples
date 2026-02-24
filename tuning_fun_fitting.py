import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# show the tuning function
kappa = 3
Amp = 30
B = 10
tf = lambda x: Amp * np.exp(kappa * np.cos(x) - kappa) + B

# simulate data
n = 12
x = np.linspace(-np.pi, np.pi - 2*np.pi/n, n)
y = tf(x) + np.random.randn(*x.shape)

# fit data with least squares
fn = lambda x, P: P[0] * np.exp(P[1] * np.cos(x) - P[1]) + P[2]
obj_fn = lambda P: np.sum((fn(x, P) - y) ** 2)
p0 = np.random.rand(3) * 3
res = minimize(obj_fn, p0)
P = res.x
print(P)

plt.figure(figsize=(8, 4))

xx = np.linspace(-np.pi, np.pi, int(1e3))
plt.plot(xx, tf(xx), 'k--')
plt.scatter(x, y)
plt.plot(xx, fn(xx, P))

plt.legend(["Ground truth", "Simulated data", "Fit"])
plt.xlabel("Orientation (2x rad.)")
plt.ylabel("Firing rate")
plt.title("Tuning function")

plt.tight_layout()
plt.show()