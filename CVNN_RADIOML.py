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
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import complextorch.nn.modules.conv as CVConv
import complextorch.nn.modules.linear as CVLinear
import complextorch.nn.modules.pooling as CVPooling
import complextorch.nn.modules.activation as CVActivation
import complextorch.nn.modules.batchnorm as CVBatchNorm
from cplxmodule import cplx
from cplxmodule.nn import CplxConv1d, CplxLinear, CplxModReLU

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
activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh", "c_elu", "c_gelu"]

data_loc = "/mnt/i/RADIOML/"

# %%
with h5py.File(data_loc + "GOLD_XYZ_OSC.0001_1024.hdf5", "r") as f:
    print(list(f.keys()))   
    x = f["X"][:1000]
    y = f["Y"][:1000]
    z = f["Z"][:1000]
    print(x.shape, y.shape, z.shape)


# %% [markdown]
# We have N frames, with 1024 time-series samples, and they are real and imaginuary, we have 2. so X will be N x 1024 x 2.
# 
# Y is the labels, one-hot encoded so we have 24 labels, only 1 of them should have 1.
# 
# Z SNR - signal to noise ratio and each series has an associated value

# %%
class ComplexRadioCNN(nn.Module):
    def __init__(self, n_classes=24):
        super().__init__()
        self.features = nn.Sequential(
            CVConv.Conv1d(1, 16, kernel_size=7, padding=3),
            # CVBatchNorm.BatchNorm1d(16),
            CVActivation.modReLU(bias=-0.1),      # <-- set bias < 0
            CVConv.Conv1d(16, 32, kernel_size=7, padding=3),
            # CVBatchNorm.BatchNorm1d(32),
            CVActivation.modReLU(bias=-0.1),
            CVPooling.AdaptiveAvgPool1d(512),   # 1024 -> 512
            CVConv.Conv1d(32, 64, kernel_size=7, padding=3),
            # CVBatchNorm.BatchNorm1d(64),
            CVActivation.modReLU(bias=-0.1),
            CVPooling.AdaptiveAvgPool1d(256),   # 512 -> 256
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            CVLinear.Linear(64 * 256, 256),
            CVActivation.modReLU(bias=-0.1),
            CVLinear.Linear(256, n_classes),
        )

    def forward(self, z):
        z = self.features(z)
        z = self.classifier(z)
        return z.abs()


# %%
X = x.astype("float32")
X_complex = X[..., 0] + 1j * X[..., 1]        # shape (N, 1024), complex64
X_complex = torch.from_numpy(X_complex).to(torch.complex64)  # (N, 1024)
X_complex = X_complex.unsqueeze(1)            # (N, 1, 1024) for Conv1d-style nets

# %%
X_real = torch.from_numpy(X).permute(0, 2, 1).float()  # (N, 2, 1024)


# %%
Y_idx = y.argmax(axis=1)          # numpy, shape (N,)
Y_idx = torch.from_numpy(Y_idx).long()

# %%
class RadioDataset(Dataset):
    def __init__(self, X_complex, Y_idx):
        self.X = X_complex   # [N,1,1024], complex
        self.y = Y_idx       # [N], long

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# %%
N = X_complex.shape[0]
split = int(0.8 * N)
train_ds = RadioDataset(X_complex[:split], Y_idx[:split])
test_ds  = RadioDataset(X_complex[split:], Y_idx[split:])

# %%

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# %%
def train_one_epoch(model, loader, optimizer, criterion, epoch, tag="Complex"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc=f"[{tag}] Train {epoch}", leave=False):
        x = x.to(device)      # complex
        y = y.to(device)      # long

        optimizer.zero_grad()
        logits = model(x)     # [B, n_classes], real
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[{tag}] Epoch {epoch} | loss={avg_loss:.4f} | acc={acc:.4f}")

# %%
def evaluate(model, loader, criterion, tag="Complex"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[{tag}] Eval", leave=False):
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
    print(f"[{tag}] Val | loss={avg_loss:.4f} | acc={acc:.4f}")
    return avg_loss, acc

# %%
model = ComplexRadioCNN(n_classes=24).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 6):
    train_one_epoch(model, train_loader, optimizer, criterion, epoch, tag="CVNN")
    evaluate(model, test_loader, criterion, tag="CVNN")

# %%
# print([n for n in dir(CVPooling)])

# %% [markdown]
# ## CPLXModule

# %%
# !pip install cplxmodule

# %%



