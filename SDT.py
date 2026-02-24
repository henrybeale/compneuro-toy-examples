import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# settings
n_trials = 500
sigma = .5  # internal noise variance
true_s = 1

# criterion in log odds of hypotheses (s=s, s=0);
B = 0  # optimal criterion is 0 for equal prior and symmetric utility

# criterion in sensory measurement space
xb = true_s/2 + (sigma**2 * B)/true_s

# probability of stimulus present (prior)
p_s = .5

x = np.full((n_trials, 1), np.nan)
r = np.full((n_trials, 1), np.nan)
s = np.full((n_trials, 1), np.nan)

# generate random samples of x given s
for n in range(n_trials):
    # sample true state, s (with prior probability)
    s[n] = np.random.choice([0, true_s], p=[1 - p_s, p_s])

    # generate internal sample, x
    x[n] = np.random.randn() * sigma + s[n]

    # threshold in x
    r[n] = x[n] >= xb

    # formula for log odds with threshold B (identical result)
    logodds = true_s * x / 2 - sigma**2 / (2 * sigma**2)

# compute d'
H = np.sum((s == 1) & (r == 1)) / np.sum(s == 1)
F = np.sum((s == 0) & (r == 1)) / np.sum(s == 0)

dprime = norm.ppf(H) - norm.ppf(F)
true_d = true_s / sigma

# compute criterion
crit = dprime/2 - norm.ppf(H)

print(f"true d': {true_d:.2f}, est.: {dprime:.2f}\ntrue B: {B:.2f}, est.: {crit:.2f}\n")

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
xx = np.linspace(-4*sigma, 4*sigma + true_s, 200)
s0 = norm.pdf(xx, 0, sigma)
s1 = norm.pdf(xx, true_s, sigma)

plt.plot(xx, s0)
plt.plot(xx, s1)
plt.axvline(xb)
plt.xlabel("x")
plt.ylabel("Density")
plt.legend(["Null distribution", "s distribution", "Optimal criterion"])
plt.title("Signal and noise distributions")

plt.subplot(1, 3, 2)
plt.bar([1, 4], [true_d, B], .2)
plt.bar([2, 5], [float(dprime), float(crit)], .2)
plt.xticks([1.5, 4.5], ["d'", "B"])
plt.xlabel("Parameter")
plt.legend(["True", "Estimated"])
plt.title("SDT parameter vs estimate")

plt.subplot(1, 3, 3)
jit = .02
xv = x.ravel()
sv = s.ravel()
rv = r.ravel()

plt.scatter(xv[(sv == true_s) & (rv == 1)], true_s + np.random.randn(np.sum((sv == true_s) & (rv == 1))) * jit, alpha=.5)
plt.scatter(xv[(sv == true_s) & (rv == 0)], true_s + np.random.randn(np.sum((sv == true_s) & (rv == 0))) * jit, alpha=.5)
plt.scatter(xv[(sv == 0) & (rv == 0)], np.random.randn(np.sum((sv == 0) & (rv == 0))) * jit, alpha=.5)
plt.scatter(xv[(sv == 0) & (rv == 1)], np.random.randn(np.sum((sv == 0) & (rv == 1))) * jit, alpha=.5)

plt.axvline(xb)
plt.legend(["Hits", "Misses", "Correct rejections", "False alarms", "Criterion"], loc="best")
plt.xlabel("x")
plt.ylabel("s (+ display jitter)")
plt.title("Internal responses by choice")

plt.tight_layout()
plt.show()