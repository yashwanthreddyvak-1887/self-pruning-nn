import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# ---------------- PRUNABLE LAYER ----------------
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights.t()) + self.bias

# ---------------- MODEL ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32*32*3, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ---------------- SPARSITY LOSS ----------------
def sparsity_loss(model):
    loss = 0
    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            loss += torch.sum(gates)
    return loss

# ---------------- TRAIN ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

lambda_sparse = 0.001

for epoch in range(3):  # keep small for fast run
    total_loss = 0
    model.train()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        ce_loss = criterion(outputs, labels)
        sp_loss = sparsity_loss(model)

        loss = ce_loss + lambda_sparse * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

# ---------------- EVALUATION ----------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print("Accuracy:", accuracy)

# ---------------- SPARSITY ----------------
total = 0
pruned = 0
all_gates = []

for module in model.modules():
    if hasattr(module, "gate_scores"):
        gates = torch.sigmoid(module.gate_scores).detach().cpu()
        total += gates.numel()
        pruned += (gates < 1e-2).sum().item()
        all_gates.extend(gates.numpy().flatten())

sparsity = (pruned / total) * 100
print("Sparsity:", sparsity)

# ---------------- PLOT ----------------
plt.hist(all_gates, bins=50)
plt.title("Gate Value Distribution")
plt.savefig("plot.png")

print("Plot saved as plot.png")
