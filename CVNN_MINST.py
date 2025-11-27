# %%
# !git clone https://github.com/torchcvnn/examples

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

# %%
# !pip install torchcvnn
import torchcvnn.nn as c_nn

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# Check torchcvnn.nn for names like ModReLU, zReLU, Cardioid, etc.

# %%
def get_complex_activation(name: str):
    name = name.lower()
    if name == "modrelu":
        return c_nn.modReLU()
    elif name == "zrelu":
        return c_nn.zReLU()
    elif name == "cardioid":
        return c_nn.Cardioid()
    elif name == "c_relu":
        return c_nn.CReLU()
    elif name == "c_sigmoid":
        return c_nn.CSigmoid()
    elif name == "c_tanh":
        return c_nn.CTanh()
    elif name == "c_elu":
        return c_nn.CELU()
    elif name == "c_gelu":
        return c_nn.CGELU()
    else:
        return nn.Identity()

# %%
class ComplexMnistCNN(nn.Module):
    def __init__(self, act_name="modrelu"):
        super().__init__()
        act = get_complex_activation(act_name)

        self.features = nn.Sequential(
            c_nn.ConvTranspose2d(1, 16, kernel_size=3, padding=1),
            c_nn.BatchNorm2d(16),
            act,
            c_nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1),
            c_nn.BatchNorm2d(32),
            act,
            c_nn.AvgPool2d(kernel_size=2, stride=2),  
            c_nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1),
            c_nn.BatchNorm2d(64),
            act,
            c_nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128, dtype=torch.complex64),
            act,
            nn.Linear(128, 10, dtype=torch.complex64),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # For classification, map complex logits to real via magnitude or real part.
        return x.abs()

# %%
class ComplexFourierMNIST(torch.utils.data.Dataset):
    def __init__(self, mnist_dataset):
        self.base = mnist_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]        # img: [1,28,28], real
        img = torch.fft.fft2(img)          # convert to complex
        img = img.to(torch.complex64)
        return img, label

# %% [markdown]
# Training Loop

# %%
def train_one_epoch(model, loader, optimizer, criterion, epoch, act_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"[{act_name}] Train {epoch}", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)            # real-valued logits from complex net
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[{act_name}] Epoch {epoch} | loss={avg_loss:.4f} | acc={acc:.4f}")

# %% [markdown]
# Eval Loop

# %%
def evaluate(model, loader, criterion, act_name):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{act_name}] Eval", leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[{act_name}] Val | loss={avg_loss:.4f} | acc={acc:.4f}")
    return avg_loss, acc

# %% [markdown]
# Main

# %%
transform = T.Compose([
    T.ToTensor(),  # [0,1] float, shape [1,28,28]
])
train_real = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_real  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# %%
train_ds = ComplexFourierMNIST(train_real)
test_ds  = ComplexFourierMNIST(test_real)

# %%
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# %%
activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh", "c_elu", "c_gelu"]

for act_name in activations_to_test:
    print(f"\n=== Testing activation: {act_name} ===")
    model = ComplexMnistCNN(act_name=act_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        train_one_epoch(model, train_loader, optimizer, criterion, epoch, act_name)
        evaluate(model, test_loader, criterion, act_name)


# %%
# print(dir(c_nn))

# %%
x0, y0 = next(iter(train_loader))
print("dataset batch dtype:", x0.dtype, x0.shape)


# %%
print(dir(c_nn))

# %%



