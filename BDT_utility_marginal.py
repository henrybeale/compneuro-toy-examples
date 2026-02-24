import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# set up a dense grid
dgrid = int(1e3)  # grid density
x = np.linspace(0, 100, dgrid)
s = np.linspace(0, 100, dgrid)

# prior over world-states, s
p_s = norm.pdf(s, 50, 5)
p_s = p_s / np.sum(p_s)

# likelihood of x for every state s
p_x_s = np.full((dgrid, dgrid), np.nan)
likfun = lambda x, s: norm.pdf(x, s, 10)
for i in range(dgrid):
    p_x_s[i, :] = likfun(x, s[i])  # [s, x]

# compute posteriors for every observed x
p_s_x = np.full((dgrid, dgrid), np.nan)
p_x = np.full((1, dgrid), np.nan)
for i in range(dgrid):
    num = p_x_s[:, i] * p_s
    p_x[0, i] = np.sum(num)  # marginal evidence for x
    p_s_x[:, i] = num / p_x[0, i]

p_x = p_x / np.sum(p_x)

# policy: apply deterministic MAP to every observed x (i.e. observer chooses
# the most probable s value based on posteriors)
a = np.full((dgrid, 1), np.nan)  # chosen state estimate for each x
for i in range(dgrid):
    max_ind = np.argmax(p_s_x[:, i])
    a[i, 0] = s[max_ind]

# reward: deterministic function of action, a, and true world-state, s.
r = -np.abs(a.T - s.reshape(-1, 1))  # distance/error [s, a(x)]

# utility is an identity function of reward (i.e. all external rewards
# treated identically by the observer)
u = r

# expected utility given policy:
E_u = np.sum(p_x * np.sum(p_s_x * r, axis=0))  # eq 2 with deterministic variables
print('Expected utility:', E_u)

plt.figure(figsize=(14, 10))

plt.subplot(2, 4, 1)
plt.plot(s, p_s)
plt.xlabel("s")
plt.title("Prior")

plt.subplot(2, 4, 2)
plt.plot(x - np.mean(x), likfun(x - np.mean(x), 0))
plt.xlabel("x - s")
plt.title("Likelihood")

plt.subplot(2, 4, 3)
plt.plot(x, p_x.ravel())
plt.xlabel("x")
plt.title("Marginal Evidence")

plt.subplot(2, 4, 4)
plt.imshow(p_x_s, aspect="auto", origin="lower",
           extent=[x[0], x[-1], s[0], s[-1]])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("s")
plt.title("Likelihoods")

plt.subplot(2, 4, 5)
plt.imshow(p_s_x, aspect="auto", origin="lower",
           extent=[x[0], x[-1], s[0], s[-1]])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("s")
plt.title("Posteriors")

plt.subplot(2, 4, 6)
plt.plot(x, a)
plt.xlabel("x")
plt.ylabel("a")
plt.title("Actions (MAP policy)")

plt.subplot(2, 4, 7)
plt.imshow(r, aspect="auto", origin="lower",
           extent=[a.min(), a.max(), s[0], s[-1]])
plt.colorbar()
plt.xlabel("a(x)")
plt.ylabel("s")
plt.title("Reward")

plt.tight_layout()
plt.show()