import numpy as np
import matplotlib.pyplot as plt

# activation function and derivative
sigm = lambda x: 1 / (1 + np.exp(-x))
dsigm = lambda x: x * (1 - x)

# training data (XOR)
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]], dtype=float)
labels = np.array([[0],
                   [1],
                   [1],
                   [0]], dtype=float)

# settings
train_steps = int(1e4)
lr = .1  # learning rate

# network size
in_size = 2
hidden_size = 2
out_size = 1

# initialise random weights and biases
L1_w = np.random.randn(in_size, hidden_size)
L1_b = np.random.randn(1, hidden_size)
out_w = np.random.randn(hidden_size, out_size)
out_b = np.random.randn(1, out_size)

# storage for history
train_error = np.full((train_steps, 1), np.nan)
train_w = np.full((train_steps, hidden_size * in_size), np.nan)
train_b = np.full((train_steps, 2), np.nan)
train_out = np.full((train_steps, 4), np.nan)

# train
for n in range(train_steps):
    # forward pass
    L1_out = sigm(inputs @ L1_w + L1_b)
    output = sigm(L1_out @ out_w + out_b)

    # error backpropagation
    error = labels - output
    d_out = error * dsigm(output)

    L1_err = d_out @ out_w.T
    d_L1 = L1_err * dsigm(L1_out)

    # update weights and biases
    out_w = out_w + lr * (L1_out.T @ d_out)
    out_b = out_b + lr * np.sum(d_out)
    L1_w = L1_w + lr * (inputs.T @ d_L1)
    L1_b = L1_b + lr * np.sum(d_L1, axis=0)

    # storage
    train_error[n, 0] = np.sqrt(np.mean(error ** 2))  # RMS
    train_w[n, :] = L1_w.reshape(-1, order="F")
    train_b[n, :] = L1_b
    train_out[n, :] = output.ravel()

print(inputs)
print(output)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_error)
plt.xlabel("Training step")
plt.ylabel("RMS error")
plt.title("Training error")

plt.subplot(2, 2, 2)
plt.plot(train_out)
plt.legend([str(int(v)) for v in labels.ravel()])
plt.xlabel("training step")
plt.ylabel("Class probability")
plt.title("Outputs")

plt.subplot(2, 2, 3)
plt.plot(train_w)
plt.xlabel("Training step")
plt.title("Layer weights")

plt.subplot(2, 2, 4)
plt.plot(train_b)
plt.title("Hidden layer biases")

plt.tight_layout()
plt.show()