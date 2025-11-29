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

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# Check torchcvnn.nn for names like ModReLU, zReLU, Cardioid, etc.

# %%
def get_complex_activation(name: str):
    name = name.lower()
    if name == "modrelu":
        return CVActivation.modReLU(bias=-0.1)
    elif name == "zrelu":
        return CVActivation.zReLU()
    elif name == "cardioid":
        return CVActivation.CVCardiod()
    elif name == "c_relu":
        return CVActivation.CReLU(inplace=False)
    elif name == "c_sigmoid":
        return CVActivation.CVSigmoid()
    elif name == "c_tanh":
        return CVActivation.CTanh()
    else:
        return nn.Identity()

# %%
data_loc = "/mnt/i/RADIOML/"

# %%
N = 200000

# %%
with h5py.File(data_loc + "GOLD_XYZ_OSC.0001_1024.hdf5", "r") as f:
    print(list(f.keys()))   
    x = f["X"][:N]
    y = f["Y"][:N]
    z = f["Z"][:N]
    print(x.shape, y.shape, z.shape)


# %% [markdown]
# We have N frames, with 1024 time-series samples, and they are real and imaginuary, we have 2. so X will be N x 1024 x 2.
# 
# Y is the labels, one-hot encoded so we have 24 labels, only 1 of them should have 1.
# 
# Z SNR - signal to noise ratio and each series has an associated value

# %%
z[0]

# %%
class ComplexRadioCNN(nn.Module):
    def __init__(self, n_classes=24, af="modrelu"):
        super().__init__()
        cvaf = get_complex_activation(af)
        
        
        self.features = nn.Sequential(
            CVConv.Conv1d(1, 16, kernel_size=7, padding=3),
            # CVBatchNorm.BatchNorm1d(16),
            cvaf,
            CVConv.Conv1d(16, 32, kernel_size=7, padding=3),
            # CVBatchNorm.BatchNorm1d(32),
            cvaf,
            CVPooling.AdaptiveAvgPool1d(512),   # 1024 -> 512
            CVConv.Conv1d(32, 64, kernel_size=7, padding=3),
            # CVBatchNorm.BatchNorm1d(64),
            cvaf,
            CVPooling.AdaptiveAvgPool1d(256),   # 512 -> 256
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            CVLinear.Linear(64 * 256, 256),
            cvaf,
            CVLinear.Linear(256, n_classes),
        )

    def forward(self, z):
        z = self.features(z)
        z = self.classifier(z)
        return z.abs()


# %%
X = x.astype("float32")
X_complex = X[..., 0] + 1j * X[..., 1]        
X_complex = torch.from_numpy(X_complex).to(torch.complex64)  
X_complex = X_complex.unsqueeze(1)            
X_real = torch.from_numpy(X).permute(0, 2, 1).float()  

Y_idx = y.argmax(axis=1)         
Y_idx = torch.from_numpy(Y_idx).long()

# %%
class RadioComplexDataset(Dataset):
    def __init__(self, X_complex, Y_idx):
        self.X = X_complex   # [N,1,1024], complex
        self.y = Y_idx       # [N], long

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# %%
# N = X_real.shape[0]
perm = torch.randperm(N)

X_real = X_real[perm]
X_complex = X_complex[perm]
Y_idx = Y_idx[perm]

split = int(0.8 * N)
cv_train_ds = RadioComplexDataset(X_complex[:split], Y_idx[:split])
cv_test_ds  = RadioComplexDataset(X_complex[split:], Y_idx[split:])
cv_train_loader = DataLoader(cv_train_ds, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True)
cv_test_loader  = DataLoader(cv_test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

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
print("X_real:", X_real.shape, X_real.dtype)       
print("X_complex:", X_complex.shape, X_complex.dtype)  
print("Y_idx:", Y_idx.shape, Y_idx.min(), Y_idx.max())  


# %%
activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh"]


for af in activations_to_test:
    print(f"\n=== Testing activation: {af} ===")


    model = ComplexRadioCNN(n_classes=24, af=af).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        train_one_epoch(model, cv_train_loader, optimizer, criterion, epoch, tag=af)
        evaluate(model, cv_test_loader, criterion, tag=af)



# %% [markdown]
# Real NN

# %%
class RadioRealDataset(Dataset):
    def __init__(self, X_real, Y_idx):
        self.X = X_real    # [N,2,1024], float32
        self.y = Y_idx     # [N], long

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

real_train_ds = RadioRealDataset(X_real[:split], Y_idx[:split])
real_test_ds  = RadioRealDataset(X_real[split:], Y_idx[split:])

real_train_loader = DataLoader(
    real_train_ds, batch_size=256, shuffle=True, num_workers=4, pin_memory=True
)
real_test_loader = DataLoader(
    real_test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
)


# %%
class RealRadioCNN(nn.Module):
    def __init__(self, n_classes=24):
        super().__init__()
        in_ch = 2  # I/Q channels

        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=7, padding=3),
            # nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size=7, padding=3),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(512),   # 1024 -> 512

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),   # 512 -> 256
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: [B,2,1024], float
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits [B, n_classes]


# %%
def real_train_one_epoch(model, loader, optimizer, criterion, epoch, tag="Real"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc=f"[{tag}] Train {epoch}", leave=False):
        x = x.to(device)    # [B,2,1024], float
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
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


def real_evaluate(model, loader, criterion, tag="Real"):
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
real_model = RealRadioCNN(n_classes=24).to(device)
real_optimizer = optim.Adam(real_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, 6):
    real_train_one_epoch(real_model, real_train_loader, real_optimizer, criterion, epoch, tag="RealCNN")
    real_evaluate(real_model, real_test_loader, criterion, tag="RealCNN")


# %%
# print([n for n in dir(CVPooling)])


