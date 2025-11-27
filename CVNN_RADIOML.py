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
import h5py
import torchcvnn.nn as c_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading
data_loc = "/mnt/i/RADIOML/"

N = 1000


with h5py.File(data_loc + "GOLD_XYZ_OSC.0001_1024.hdf5", "r") as f:
    print(list(f.keys()))   
    x = f["X"][:N]   # N x 1024 x 2
    y = f["Y"][:N]   # N x 1024 x 2
    z = f["Z"][:N]   # N x 1024 x 2
    print(x.shape, y.shape, z.shape)

# %%
class ComplexRadiolMLDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __len__(self):  # N
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

# %%
train_ds = ComplexRadiolMLDataset(x, y, z)
test_ds = ComplexRadiolMLDataset(x, y, z)

# %%
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)        


class RealCNN(nn.Module):
    def __init__(self, use_two_channels=False):
        super().__init__()
        in_ch = 2 if use_two_channels else 1

        self.features = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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

class ComplexCNN(nn.Module):
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


# ----- TO DO -----
# 1. Data loading and Processing to enter the CNN
# 2. Complex CNN
# 3. Real CNN
# 4. Training loop
# 5. Evaluation loop
# 6. Real vs Complex

# ----- TO DO -----
