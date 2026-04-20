import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
LAMBDAS = [1e-4, 5e-4, 1e-3]
THRESHOLD = 5e-2

# =========================
# CUSTOM PRUNABLE LINEAR LAYER
# =========================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -2.0))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)

# =========================
# MODEL
# =========================
class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sparsity_loss(self):
        loss = 0
        for layer in [self.fc1, self.fc2, self.fc3]:
            loss += layer.get_gates().sum()
        return loss

    def all_gates(self):
        gates = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates.append(layer.get_gates().detach().cpu().view(-1))
        return torch.cat(gates)

# =========================
# DATA
# =========================
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# EVALUATE FUNCTION
# =========================
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# =========================
# SPARSITY FUNCTION
# =========================
def calculate_sparsity(model, threshold=THRESHOLD):
    total = 0
    pruned = 0

    for layer in [model.fc1, model.fc2, model.fc3]:
        gates = layer.get_gates().detach()
        total += gates.numel()
        pruned += (gates < threshold).sum().item()

    return 100 * pruned / total

# =========================
# TRAIN FUNCTION
# =========================
def train_model(lambda_sparse):
    model = SelfPruningNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            classification_loss = criterion(outputs, labels)
            sparse_loss = model.sparsity_loss()
            total_loss = classification_loss + lambda_sparse * sparse_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        acc = evaluate_model(model)
        sparsity = calculate_sparsity(model, threshold=THRESHOLD)
        print(
            f"Lambda={lambda_sparse} | Epoch {epoch+1}/{EPOCHS} | "
            f"Loss={running_loss:.4f} | Test Acc={acc:.2f}% | Sparsity={sparsity:.2f}%"
        )

    final_acc = evaluate_model(model)
    return model, final_acc

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    results = []
    best_overall_model = None
    best_overall_lambda = None
    best_overall_acc = 0

    for lam in LAMBDAS:
        print("\n" + "=" * 60)
        print(f"Training with lambda = {lam}")
        print("=" * 60)

        model, acc = train_model(lam)
        sparsity = calculate_sparsity(model)

        results.append((lam, acc, sparsity))
        print(f"Final Result | Lambda={lam} | Accuracy={acc:.2f}% | Sparsity={sparsity:.2f}%")

        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_model = model
            best_overall_lambda = lam

    print("\n===== FINAL RESULTS =====")
    print(f"{'Lambda':<10} {'Accuracy (%)':<15} {'Sparsity (%)':<15}")
    for lam, acc, sparsity in results:
        print(f"{lam:<10} {acc:<15.2f} {sparsity:<15.2f}")

    best_gates = best_overall_model.all_gates().numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(best_gates, bins=50)
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.title(f"Distribution of Final Gate Values (Best Lambda = {best_overall_lambda})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gate_distribution.png")
    plt.show()