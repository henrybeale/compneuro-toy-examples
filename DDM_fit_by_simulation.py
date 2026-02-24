import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pybads import BADS

# fit the DDM using simulation and comparison of CDFs (signed RT)

# you must install BADS: https://github.com/acerbilab/pybads 

# generate some data
n_trials = 500
true_v = .8
true_a = 1
true_z = .5
true_t = .2
signed_rt = None  # filled after sim_ddm is defined

# compute cumulative density over reasonable range
xx = np.linspace(-3, 3, 1000)

def sim_ddm(n_trials, v, a, z, t):
    # function to simulate DDM trials (see DDM.m)
    s = 1
    dt = .001
    time = np.arange(0, 25 + dt, dt)
    signed_rt = np.full((n_trials, 1), np.nan)

    for n in range(n_trials):
        x = z * a
        for j in range(1, time.size):
            x = x + v * dt + s * np.sqrt(dt) * np.random.randn()
            if x >= a or x <= 0:
                signed_rt[n, 0] = t + (j + 1) * dt
                if x <= 0:
                    signed_rt[n, 0] = -1 * signed_rt[n, 0]
                break

        if j == time.size - 1:
            rt = np.inf
            choice = -1

    return signed_rt

signed_rt = sim_ddm(n_trials, true_v, true_a, true_z, true_t)

get_cdf = lambda y: (y <= xx.reshape(1, -1)).mean(axis=0)
cdf_data = get_cdf(signed_rt)

# wrap simulation, cdf, and distance from data cdf
n_sim_trials = int(5e4)  # increase for accuracy > compute time
cdf_sim = lambda P: get_cdf(sim_ddm(n_sim_trials, P[0], P[1], P[2], P[3]))
fn = lambda P: np.mean((cdf_data - cdf_sim(P)) ** 2)

# optimise using BADS function
lb = np.array([.1, 0, 0, .11])
ub = np.array([3, 3, 1, .8])
plb = np.array([.2, .3, .35, .18])  # plausible lower/upper bounds
pub = np.array([1., 1.5, .65, .5])

p0 = np.random.rand(lb.size) * (pub - plb) + plb  # start parameters
bads = BADS(fn, p0, lb, ub, plb, pub)
P = bads.optimize()
print(P)

srt_fit = sim_ddm(n_sim_trials, P[0], P[1], P[2], P[3])
cdf_fit = get_cdf(srt_fit)

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.plot(xx, cdf_data)
plt.plot(xx, cdf_fit)
plt.legend(["Data", "Fitted"])
plt.xlabel("Signed RT")
plt.ylabel("Probability of observation")
plt.title("Cumulative distributions")

plt.subplot(1, 3, 2)
kde_data = gaussian_kde(signed_rt[np.isfinite(signed_rt)].ravel())
plt.plot(xx, kde_data(xx))
kde_fit = gaussian_kde(srt_fit[np.isfinite(srt_fit)].ravel())
plt.plot(xx, kde_fit(xx))
plt.legend(["Data", "Fitted"])
plt.xlabel("Signed RT")
plt.ylabel("Density")
plt.title("Smoothed distributions")

plt.subplot(1, 3, 3)
plt.bar((np.arange(1, 5)) - .1, [true_v, true_a, true_z, true_t], .2)
plt.bar((np.arange(1, 5)) + .1, P, .2)
plt.xticks(np.arange(1, 5), ["v", "a", "z", "t"])
plt.xlabel("Parameter")
plt.legend(["True", "Fitted"])

plt.tight_layout()
plt.show()