import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

burnin = 5e2
max_steps = int(burnin + 5e3)

# proposal distribution
def g(x):
    return np.random.randn() * 1 + x  # N(x, 1)

# desired distribution: 0.6*N(10,3) + 0.4*N(20,5)
def P(x):
    return 0.6 * norm.pdf(x, loc=10, scale=3) + 0.4 * norm.pdf(x, loc=20, scale=5)

X = np.full(max_steps, np.nan, dtype=float)
X[0] = np.random.randn()  # random initialisation

for t in range(1, max_steps):
    # proposal
    xp = g(X[t - 1])

    # compute product terms (use log)
    # add tiny epsilon to avoid log(0) if P underflows
    eps = 1e-300
    a1 = np.log(P(xp) + eps) - np.log(P(X[t - 1]) + eps)

    # for symmetric normal proposal, this cancels to 0; kept for fidelity
    a2 = norm.logpdf(X[t - 1], loc=xp, scale=1) - norm.logpdf(xp, loc=X[t - 1], scale=1)

    # acceptance probability
    alpha = min(1.0, float(np.exp(a1 + a2)))
    if np.random.rand() <= alpha:
        X[t] = xp
    else:
        X[t] = X[t - 1]

# ---- plots (two "tiles") ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

burnin_i = int(burnin)

ax1.plot(np.arange(1, burnin_i + 1), X[:burnin_i], label="Burn-in")
ax1.plot(np.arange(burnin_i + 1, max_steps + 1), X[burnin_i:], label="Samples")
ax1.set_xlabel("Step")
ax1.set_ylabel("x")
ax1.legend()

samples = X[burnin_i:]
kde = gaussian_kde(samples)
xs = np.linspace(samples.min() - 3 * samples.std(), samples.max() + 3 * samples.std(), 500)

ax2.plot(xs, kde(xs), label="Sampled")
ax2.plot(xs, P(xs), "k--", label="Target")
ax2.set_xlabel("s")
ax2.set_ylabel("p(s)")
ax2.legend()

plt.tight_layout()
plt.show()