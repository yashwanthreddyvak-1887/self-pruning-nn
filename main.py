import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------- SIMPLE DATA ----------------
# Fake dataset (since no PyTorch)
X = np.random.rand(500, 5)
y = (np.sum(X, axis=1) > 2.5).astype(int)

# ---------------- MODEL ----------------
class PrunableLayer:
    def __init__(self, in_f, out_f):
        self.weight = np.random.randn(in_f, out_f)
        self.bias = np.zeros(out_f)
        self.gate_scores = np.random.randn(in_f, out_f)

    def forward(self, x):
        gates = 1 / (1 + np.exp(-self.gate_scores))  # sigmoid
        pruned_weights = self.weight * gates
        return np.dot(x, pruned_weights) + self.bias, gates

# ---------------- NETWORK ----------------
class Net:
    def __init__(self):
        self.layer1 = PrunableLayer(5, 4)
        self.layer2 = PrunableLayer(4, 1)

    def forward(self, x):
        out1, g1 = self.layer1.forward(x)
        out1 = np.maximum(0, out1)  # ReLU
        out2, g2 = self.layer2.forward(out1)
        return out2, [g1, g2]

# ---------------- LOSS ----------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_loss(pred, target, gates, lambda_sparse=0.01):
    pred = sigmoid(pred).reshape(-1)
    ce_loss = -np.mean(target*np.log(pred+1e-8) + (1-target)*np.log(1-pred+1e-8))

    sparsity = sum(np.sum(g) for g in gates)
    return ce_loss + lambda_sparse * sparsity

# ---------------- TRAIN ----------------
model = Net()
lr = 0.01

for epoch in range(50):
    pred, gates = model.forward(X)
    loss = compute_loss(pred, y, gates)

    # fake gradient update (simple simulation)
    for layer in [model.layer1, model.layer2]:
        layer.weight -= lr * np.random.randn(*layer.weight.shape)
        layer.gate_scores -= lr * np.random.randn(*layer.gate_scores.shape)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---------------- EVALUATION ----------------
pred, gates = model.forward(X)
pred = sigmoid(pred).reshape(-1)
pred_labels = (pred > 0.5).astype(int)

accuracy = np.mean(pred_labels == y) * 100
print("Accuracy:", accuracy)

# ---------------- SPARSITY ----------------
total = 0
pruned = 0

for g in gates:
    total += g.size
    pruned += np.sum(g < 0.1)

sparsity = (pruned / total) * 100
print("Sparsity:", sparsity)

# ---------------- PLOT ----------------
all_gates = np.concatenate([g.flatten() for g in gates])

plt.hist(all_gates, bins=30)
plt.title("Gate Distribution")
plt.savefig("plot.png")

print("Done. Plot saved as plot.png")