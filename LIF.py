import numpy as np
import matplotlib.pyplot as plt

# parameters
C = 1
R = 10
theta = -55
u_reset = -80
u_0 = -70
sigma = 0.5  # noise variance (as in your MATLAB comment)

# time
fs = 1024 # sample rate
dt = 1 / fs * 1e3  # ms
time = np.arange(0, 200 + dt, dt)  

# input stimulus
I = 2 * ((time > 10) & (time < 110)).astype(float)

u = np.zeros_like(time) + u_0
spikes = np.zeros_like(time)

for t in range(len(time) - 1):
    if u[t] >= theta:
        u[t + 1] = u_reset
        spikes[t] = 1
    else:
        du = ((u_0 - u[t]) / R + I[t]) / C + np.random.randn() * sigma
        u[t + 1] = u[t] + dt * du

spike_times = np.where(spikes == 1)[0]

# plotting
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# First tile: membrane potential + spike time markers
axes[0].plot(time, u)
for st in time[spikes == 1]:
    axes[0].axvline(st, linestyle='--', linewidth=0.8)
axes[0].set_ylabel('u')
axes[0].set_title('Membrane potential')

# Second tile: input current
axes[1].plot(time, I)
axes[1].set_ylabel('I')
axes[1].set_xlabel('Time (ms)')
axes[1].set_title('Input stimulus')

plt.tight_layout()
plt.show()

