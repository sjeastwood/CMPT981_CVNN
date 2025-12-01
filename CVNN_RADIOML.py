import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
from cplxmodule.nn import CplxModReLU

import torchcvnn.nn as c_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SafeCReLU(nn.Module):
    def forward(self, z):
        return torch.complex(F.relu(z.real), F.relu(z.imag))

def get_complex_activation(name: str):
    name = name.lower()
    if name == "modrelu":
        return CVActivation.modReLU(bias=-0.1)
    elif name == "zrelu":
        return CVActivation.zReLU()
    elif name == "cardioid":
        return CVActivation.CVCardiod()
    elif name == "c_relu":
        return SafeCReLU()
    elif name == "c_sigmoid":
        return CVActivation.CVSigmoid()
    elif name == "c_tanh":
        return CVActivation.CTanh()
    else:
        return nn.Identity()

def get_datasets_snr(snr=0):
    data_path = os.path.join("data", "RadioML", "GOLD_XYZ_Subset.hdf5")
    datasets = {}
    with h5py.File(data_path, "r") as f:
        if isinstance(snr, list):
            if len(snr) == 0:
                for z in range(-20, 32, 2):
                    datasets[z] = {
                        "X": f[f"X_SNR_{z}"][:],
                        "Y": f[f"Y_SNR_{z}"][:],
                        "Z": f[f"Z_SNR_{z}"][:],
                    }
            else:
                for z in snr:
                    datasets[z] = {
                        "X": f[f"X_SNR_{z}"][:],
                        "Y": f[f"Y_SNR_{z}"][:],
                        "Z": f[f"Z_SNR_{z}"][:],
                    }
        elif isinstance(snr, int):
            datasets[snr] = {
                "X": f[f"X_SNR_{snr}"][:],
                "Y": f[f"Y_SNR_{snr}"][:],
                "Z": f[f"Z_SNR_{snr}"][:],
            }
    return datasets

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
    

dataset = get_datasets_snr()

x = dataset[0]["X"]
y = dataset[0]["Y"]
z = dataset[0]["Z"]

X = x.astype("float32")
X_complex = X[..., 0] + 1j * X[..., 1]        # shape (N, 1024), complex64
X_complex = torch.from_numpy(X_complex).to(torch.complex64)  # (N, 1024)
X_complex = X_complex.unsqueeze(1)            # (N, 1, 1024) for Conv1d-style nets

X_real = torch.from_numpy(X).permute(0, 2, 1).float()  # (N, 2, 1024)
Y_idx = y.argmax(axis=1)          # numpy, shape (N,)
Y_idx = torch.from_numpy(Y_idx).long()

class RadioDataset(Dataset):
    def __init__(self, X_complex, Y_idx):
        self.X = X_complex   # [N,1,1024], complex
        self.y = Y_idx       # [N], long

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

N = X_complex.shape[0]
split = int(0.8 * N)
train_ds = RadioDataset(X_complex[:split], Y_idx[:split])
test_ds  = RadioDataset(X_complex[split:], Y_idx[split:])

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_one_epoch(model, loader, optimizer, criterion, epoch, tag="Complex"):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    batch_grad_norms = []

    for x, y in tqdm(loader, desc=f"[{tag}] Train {epoch}", leave=False):
        x = x.to(device)      # complex
        y = y.to(device)      # long

        optimizer.zero_grad()
        logits = model(x)     # [B, n_classes], real
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = get_grad_norm(model)
        batch_grad_norms.append(grad_norm)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    avg_grad_norm = np.mean(batch_grad_norms) if batch_grad_norms else 0.0
    print(f"[{tag}] Epoch {epoch} | loss={avg_loss:.4f} | acc={acc:.4f} | avg_norm={avg_grad_norm:.4f}")

    return avg_loss, acc, avg_grad_norm

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

def run_grid_search(learning_rates, activations, epochs=5):
    grid_results = {}
    activations = activations if isinstance(activations, list) else [activations]
    learning_rates = learning_rates if isinstance(learning_rates, list) else [learning_rates]
    for af in activations:
        grid_results[af] = {}
        
        for lr in learning_rates:
            run_metrics = {
                "train_loss": [], "val_loss": [],
                "train_acc": [], "val_acc": [],
                "grad_norm": []
            }

            model = ComplexRadioCNN(n_classes=24, af=af).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(1, epochs + 1):
                
                tag_str = f"{af}|lr={lr}"
                
                t_loss, t_acc, g_norm = train_one_epoch(model, train_loader, optimizer, criterion, epoch, tag=tag_str)
                v_loss, v_acc = evaluate(model, test_loader, criterion, tag=tag_str)
                
                # Log metrics
                run_metrics["train_loss"].append(t_loss)
                run_metrics["train_acc"].append(t_acc)
                run_metrics["grad_norm"].append(g_norm)
                run_metrics["val_loss"].append(v_loss)
                run_metrics["val_acc"].append(v_acc)

            # Store run
            grid_results[af][lr] = run_metrics

    return grid_results

def plot_grid_results(grid_results, activations, learning_rates):
    metrics_map = [
        ("train_loss", "Training Loss"),
        ("val_loss",   "Validation Loss"),
        ("train_acc",  "Training Accuracy"),
        ("val_acc",    "Validation Accuracy"),
        ("grad_norm",  "Gradient Norm (L2)")
    ]
    
    # 1. Iterate over Learning Rates (Create 1 Figure per LR)
    for lr in learning_rates:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f"Performance @ Learning Rate = {lr}", fontsize=16)
        
        # 2. Iterate over Metrics (Create 5 Subplots)
        for ax_idx, (metric_key, title) in enumerate(metrics_map):
            ax = axes[ax_idx]
            
            # 3. Iterate over Activations (Plot lines on current Subplot)
            for af in activations:
                if af in grid_results and lr in grid_results[af]:
                    data = grid_results[af][lr][metric_key]
                    epochs = range(1, len(data) + 1)
                    
                    label = f"{af}"
                    ax.plot(epochs, data, marker='.', label=label)
            
            # Formatting (Must happen inside the metric loop)
            ax.set_title(title)
            ax.set_xlabel("Epochs")
            ax.set_ylabel(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend to the first subplot to avoid clutter
            if ax_idx == 0:
                ax.legend(fontsize='small', loc='best')
            
        plt.tight_layout()
        
        # Save with unique filename so LRs don't overwrite each other
        folder = "graph"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/grid_search_results_lr_{lr}.png"
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.show()


test_activations = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh"]
test_lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6] # Grid search values
epochs = 5

results = run_grid_search(test_lrs, test_activations, epochs=5)

print("Plotting results...")
plot_grid_results(results, test_activations, test_lrs)











# for af in activations_to_test:
#     grid_results[af] = {}
#     print(f"\n==========================================")
#     print(f" Activation: {af}")
#     print(f"==========================================")
    
#     for lr in learning_rates:
#         print(f"--> Grid: LR = {lr}")
        
#         # 1. Initialize storage for this run
#         run_metrics = {
#             "train_loss": [], "val_loss": [],
#             "train_acc": [], "val_acc": [],
#             "grad_norm": []
#         }
        
#         # 2. Re-Initialize Model & Optimizer (Crucial for Grid Search)
#         model = ComplexRadioCNN(n_classes=24, af=af).to(device)
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#         criterion = nn.CrossEntropyLoss()

#         # 3. Training Loop
#         for epoch in range(1, num_epochs + 1):
#             tag_str = f"{af}|lr={lr}"
            
#             t_loss, t_acc, g_norm = train_one_epoch(model, train_loader, optimizer, criterion, epoch, tag=tag_str)
#             v_loss, v_acc = evaluate(model, test_loader, criterion, tag=tag_str)
            
#             # Log metrics
#             run_metrics["train_loss"].append(t_loss)
#             run_metrics["train_acc"].append(t_acc)
#             run_metrics["grad_norm"].append(g_norm)
#             run_metrics["val_loss"].append(v_loss)
#             run_metrics["val_acc"].append(v_acc)
            
#             print(f"    Ep {epoch}: TrainLoss={t_loss:.4f}, ValAcc={v_acc:.4f}, GradNorm={g_norm:.2f}")

#         # Store run
#         grid_results[af][lr] = run_metrics


