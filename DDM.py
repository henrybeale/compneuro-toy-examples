import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# simulate from the ddm
n_trials = 10000
rt = np.full((n_trials, 1), np.nan)
choice = np.full((n_trials, 1), np.nan)

# parameters
v = .8  # drift rate
a = 1   # threshold
z = .5  # starting point bias
t = .2  # non-decision time
s = 1   # noise variance

# time vector for simulation
dt = .001
time = np.arange(0, 25 + dt, dt)  # seconds

x = np.full((n_trials, time.size), np.nan)
for n in range(n_trials):
    # start point
    x[n, 0] = z * a

    for j in range(1, time.size):
        # cumulative evidence
        x[n, j] = x[n, j - 1] + v * dt + s * np.sqrt(dt) * np.random.randn()

        # threshold crossin
        if x[n, j] >= a or x[n, j] <= 0:
            choice[n] = 1
            rt[n] = t + dt * (j + 1)  # add non-decision time
            if x[n, j] <= 0:
                choice[n] = choice[n] * -1
            break

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.axhline(0)
plt.axhline(a)
for n in range(10):
    plt.plot(time[time < rt[n]], x[n, time < rt[n]])
    plt.scatter(rt[n] - t, a * (choice[n] + 1) / 2, c="k")
plt.xlabel("Decision time")
plt.ylabel("Evidence")
plt.title("Simulated trials")

plt.subplot(1, 2, 2)
rdat = rt * choice

xx = np.linspace(-3, 3, 1000)
kde = gaussian_kde(rdat[np.isfinite(rdat)].ravel())
pdf = kde(xx)
plt.plot(xx, pdf)
plt.xlabel("Signed reaction time")
plt.ylabel("Density")
plt.title("RT-choice distribution")

plt.tight_layout()
plt.show()