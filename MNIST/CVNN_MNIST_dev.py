import os, time
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

import torchcvnn.nn as c_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

class RealMnistCNN(nn.Module):
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

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise

def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def train_one_epoch(model, loader, optimizer, criterion, epoch, act_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_norm = 0.0

    pbar = tqdm(loader, desc=f"[{act_name}] Train {epoch}", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)               # real-valued logits from complex net
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = compute_grad_norm(model)
        total_grad_norm += grad_norm
        
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    avg_grad_norm = total_grad_norm / len(loader) # Average over batches
    
    print(f"[{act_name}] Epoch {epoch} | loss={avg_loss:.4f} | acc={acc:.4f} | grad_norm={avg_grad_norm:.4f}")
    
    return avg_loss, acc, avg_grad_norm

def real_train_one_epoch(real_model, loader, optimizer, criterion, epoch):
    real_model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_norm = 0.0

    pbar = tqdm(loader, desc=f"[Real CNN Train {epoch}", leave=False)
    for x, y in pbar:
        xr = torch.view_as_real(x)            # [B,1,28,28,2]
        xr = xr.squeeze(1)                    # [B,28,28,2]
        xr = xr.permute(0, 3, 1, 2)           # [B,2,28,28]
        xr = xr.float().to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = real_model(xr)
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = compute_grad_norm(real_model)
        total_grad_norm += grad_norm

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    avg_grad_norm = total_grad_norm / len(loader)

    print(f"[Real CNN] Epoch {epoch} | loss={avg_loss:.4f} | acc={acc:.4f} | grad_norm={avg_grad_norm:.4f}")
    
    return avg_loss, acc, avg_grad_norm


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

def real_evaluate(real_model, loader, criterion):
    real_model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[Real CNN Eval", leave=False)
        for x, y in pbar:
            xr = torch.view_as_real(x)        # [B,1,28,28,2]
            xr = xr.squeeze(1)                # [B,28,28,2]
            xr = xr.permute(0, 3, 1, 2)       # [B,2,28,28]
            xr = xr.float().to(device)

            y = y.to(device)

            logits = real_model(xr)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[Real CNN] | loss={avg_loss:.4f} | acc={acc:.4f}")
    return avg_loss, acc

def plot_comparison(history):
    sns.set_theme(style="whitegrid")
    
    # Define the 6 metrics to plot
    metrics = ["train_loss", "train_acc", "val_loss", "val_acc", "grad_norm", "epoch_times"]
    titles = ["Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy", "Gradient Norm", "Time per Epoch (s)"]
    
    # Create a 2x3 layout (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten() # Flatten 2D array to 1D for easy iteration
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for model_name, stats in history.items():
            if metric in stats:
                # Plot data points with markers for clarity
                ax.plot(stats[metric], label=model_name, linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel("Epochs")
        
        # if "acc" in metric:
        #     ax.set_ylim(0, 1.05)
            
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.tight_layout()
    plt.show()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/mnist_comparison.png", bbox_inches='tight')


transform = T.Compose([
    T.ToTensor(),
    AddGaussianNoise(0., 0.2),    # <--- NEW: Add Gaussian Noise (mean=0, std=0.2)
    T.Lambda(lambda x: torch.clamp(x, 0, 1)),  # [0,1] float, shape [1,28,28]
])

train_real = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_real  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_ds = ComplexFourierMNIST(train_real)
test_ds  = ComplexFourierMNIST(test_real)


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)


# def show_batch(loader):
#     # Get a batch
#     images_complex, _ = next(iter(loader))
    
#     # Inverse FFT to visualize what the noisy spatial image looks like
#     # images are [B, 28, 28] complex64
#     images_spatial = torch.fft.ifft2(images_complex)
#     images_spatial = images_spatial.real # Take real part for visualization
    
#     grid = torchvision.utils.make_grid(images_spatial.unsqueeze(1)[:16], nrow=4)
#     plt.figure(figsize=(6,6))
#     plt.imshow(grid.permute(1, 2, 0))
#     plt.title("Noisy Inputs (Reconstructed from FFT)")
#     plt.axis('off')
#     plt.show()

# print("Visualizing noisy data...")
# show_batch(train_loader)


history = {}
num_epochs = 5

# activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh", "c_elu", "c_gelu"]
activations_to_test = ["modrelu", "c_relu", "c_tanh"]

for act_name in activations_to_test:
    print(f"\n=== Training activation: {act_name} ===")
    
    history[act_name] = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": [], 
        "grad_norm": [], "epoch_times": [] # <--- New List
    }
    
    model = ComplexMnistCNN(act_name=act_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        # <--- Start Timer
        start_time = time.time()
        
        t_loss, t_acc, g_norm = train_one_epoch(model, train_loader, optimizer, criterion, epoch, act_name)
        v_loss, v_acc = evaluate(model, test_loader, criterion, act_name)
        
        # <--- End Timer
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Store data
        history[act_name]["train_loss"].append(t_loss)
        history[act_name]["train_acc"].append(t_acc)
        history[act_name]["grad_norm"].append(g_norm)
        history[act_name]["val_loss"].append(v_loss)
        history[act_name]["val_acc"].append(v_acc)
        history[act_name]["epoch_times"].append(elapsed) # <--- Store Time

# --- 3. Train Real Model (Baseline) ---
print(f"\n=== Training Real CNN Baseline ===")
real_model_name = "Real CNN"
history[real_model_name] = {
    "train_loss": [], "train_acc": [], 
    "val_loss": [], "val_acc": [], 
    "grad_norm": [], "epoch_times": [] # <--- New List
}

real_model = RealMnistCNN(use_two_channels=True).to(device)
optimizer = optim.Adam(real_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, num_epochs + 1):
    # <--- Start Timer
    start_time = time.time()
    
    t_loss, t_acc, g_norm = real_train_one_epoch(real_model, train_loader, optimizer, criterion, epoch)
    v_loss, v_acc = real_evaluate(real_model, test_loader, criterion)
    
    # <--- End Timer
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Store data
    history[real_model_name]["train_loss"].append(t_loss)
    history[real_model_name]["train_acc"].append(t_acc)
    history[real_model_name]["grad_norm"].append(g_norm)
    history[real_model_name]["val_loss"].append(v_loss)
    history[real_model_name]["val_acc"].append(v_acc)
    history[real_model_name]["epoch_times"].append(elapsed) # <--- Store Time

# --- 4. Plot Everything ---
print("\nGenerating Performance Graphs...")
plot_comparison(history)





