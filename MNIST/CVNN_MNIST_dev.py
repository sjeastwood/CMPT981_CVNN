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
from torch.distributions.multivariate_normal import MultivariateNormal
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
    def __init__(self, mnist_dataset, transform=None, complex_noise=False):
        self.base = mnist_dataset
        self.transform = transform
        self.complex_noise = complex_noise

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]        # img: [1,28,28], real
        img = torch.fft.fft2(img)          # convert to complex
        img = img.to(torch.complex64)
        if self.complex_noise:
            img = self.transform(img)
        return img, label

class AddRealNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0, 1)

class AddComplexNoise(object):
    def __init__(self, mean=0.+0.j, std=0.1):
        self.mean = torch.tensor([mean.real, mean.imag])
        var = (std**2) / 2
        self.cov = torch.tensor([
            [var, 0.0], 
            [0.0, var]
        ])
        
    def __call__(self, tensor):
        tensor_view = torch.view_as_real(tensor)
        dist = MultivariateNormal(
            loc=self.mean.to(tensor.device), 
            covariance_matrix=self.cov.to(tensor.device)
        )
        noise = dist.sample(sample_shape=tensor.shape)
        noisy_view = tensor_view + noise
        return torch.view_as_complex(noisy_view)
        



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

def plot_comparison(history, epochs, lr):
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
                ax.plot(range(1, epochs+1), stats[metric], label=model_name, linewidth=2, marker='o', markersize=3, alpha=0.8)
        
        ax.set_title(titles[i], fontsize=12, fontweight='bold')
        ax.set_xlabel("Epochs")

        if "loss" in metric or "grad_norm" in metric:
            ax.set_yscale('log')
            # Add grid for minor ticks for better readability on log scale
            ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # if "acc" in metric:
        #     ax.set_ylim(0, 1.05)
            
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.suptitle(f"Comparisons under learning rate = {lr}")

    plt.tight_layout()
    plt.show()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/mnist_comparison_{lr}.png", bbox_inches='tight')


def experiment_lr(lr, num_epochs, activations, train_loader, test_loader):
    local_history = {}
    print(f"\n{'='*20}\nRunning Experiment with LR = {lr}\n{'='*20}")

    for act_name in activations:
        model_key = f"{act_name} (lr={lr})"
        print(f"--- Training {model_key} ---")
        
        local_history[model_key] = {
            "train_loss": [], "train_acc": [], 
            "val_loss": [], "val_acc": [], 
            "grad_norm": [], "epoch_times": []
        }
        
        model = ComplexMnistCNN(act_name=act_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            t_loss, t_acc, g_norm = train_one_epoch(model, train_loader, optimizer, criterion, epoch, model_key)
            v_loss, v_acc = evaluate(model, test_loader, criterion, model_key)
            elapsed = time.time() - start
            
            local_history[model_key]["train_loss"].append(t_loss)
            local_history[model_key]["train_acc"].append(t_acc)
            local_history[model_key]["grad_norm"].append(g_norm)
            local_history[model_key]["val_loss"].append(v_loss)
            local_history[model_key]["val_acc"].append(v_acc)
            local_history[model_key]["epoch_times"].append(elapsed)

    model_key = f"Real CNN (lr={lr})"
    print(f"--- Training {model_key} ---")
    
    local_history[model_key] = {
        "train_loss": [], "train_acc": [], 
        "val_loss": [], "val_acc": [], 
        "grad_norm": [], "epoch_times": []
    }

    real_model = RealMnistCNN(use_two_channels=True).to(device)
    optimizer = optim.Adam(real_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        t_loss, t_acc, g_norm = real_train_one_epoch(real_model, train_loader, optimizer, criterion, epoch)
        v_loss, v_acc = real_evaluate(real_model, test_loader, criterion)
        elapsed = time.time() - start

        local_history[model_key]["train_loss"].append(t_loss)
        local_history[model_key]["train_acc"].append(t_acc)
        local_history[model_key]["grad_norm"].append(g_norm)
        local_history[model_key]["val_loss"].append(v_loss)
        local_history[model_key]["val_acc"].append(v_acc)
        local_history[model_key]["epoch_times"].append(elapsed)

    return local_history


    history = {}
    
    print(f"\n{'='*60}")
    print(f"COMPARATIVE GRID SEARCH: {target_act} vs Real CNN")
    print(f"Testing LRs: {lr_list}")
    print(f"{'='*60}")

    for lr in lr_list:
        complex_key = f"{target_act} (lr={lr})"
        print(f"   Training {complex_key}...")
        
        history[complex_key] = {
            "train_loss": [], "train_acc": [], 
            "val_loss": [], "val_acc": [], 
            "grad_norm": [], "epoch_times": []
        }
        
        c_model = ComplexMnistCNN(act_name=target_act).to(device)
        c_opt = optim.Adam(c_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            tl, ta, gn = train_one_epoch(c_model, train_loader, c_opt, criterion, epoch, complex_key)
            vl, va = evaluate(c_model, test_loader, criterion, complex_key)
            elapsed = time.time() - start
            
            history[complex_key]["train_loss"].append(tl)
            history[complex_key]["train_acc"].append(ta)
            history[complex_key]["grad_norm"].append(gn)
            history[complex_key]["val_loss"].append(vl)
            history[complex_key]["val_acc"].append(va)
            history[complex_key]["epoch_times"].append(elapsed)

        # --- B. Train Real Baseline ---
        real_key = f"Real CNN (lr={lr})"
        print(f"   Training {real_key}...")

        history[real_key] = {
            "train_loss": [], "train_acc": [], 
            "val_loss": [], "val_acc": [], 
            "grad_norm": [], "epoch_times": []
        }

        r_model = RealMnistCNN(use_two_channels=True).to(device)
        r_opt = optim.Adam(r_model.parameters(), lr=lr)
        
        for epoch in range(1, num_epochs + 1):
            start = time.time()
            tl, ta, gn = real_train_one_epoch(r_model, train_loader, r_opt, criterion, epoch)
            vl, va = real_evaluate(r_model, test_loader, criterion)
            elapsed = time.time() - start

            history[real_key]["train_loss"].append(tl)
            history[real_key]["train_acc"].append(ta)
            history[real_key]["grad_norm"].append(gn)
            history[real_key]["val_loss"].append(vl)
            history[real_key]["val_acc"].append(va)
            history[real_key]["epoch_times"].append(elapsed)

    return history

def visualize_single_sample(real_ds, complex_ds, idx=0):
    # 1. Get Samples
    real_img, real_label = real_ds[idx]       # Shape: [1, 28, 28]
    comp_img, comp_label = complex_ds[idx]    # Shape: [1, 28, 28] (Complex64)

    # 2. Process Real Image
    # Remove channel dim for plotting: [28, 28]
    real_plot = real_img.squeeze().numpy()

    # 3. Process Complex Image (Frequency Domain)
    # We use fftshift to move the low frequencies (DC component) to the center of the image
    comp_shifted = torch.fft.fftshift(comp_img.squeeze())
    
    # A. Magnitude Spectrum (Log scale is standard because DC component is massive)
    magnitude = comp_shifted.abs()
    log_magnitude = torch.log(1 + magnitude).numpy()
    
    # B. Phase Spectrum
    phase = comp_shifted.angle().numpy()
    
    # C. Reconstructed (Inverse FFT check)
    # We reverse the FFT to see if it looks like the original
    reconstructed = torch.fft.ifft2(comp_img).real.squeeze().numpy()

    # 4. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot 1: Original Real Input
    im1 = axes[0].imshow(real_plot, cmap='gray')
    axes[0].set_title(f"Real Spatial Input\nLabel: {real_label}")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot 2: Frequency Magnitude (Log Scale)
    im2 = axes[1].imshow(log_magnitude, cmap='inferno')
    axes[1].set_title("Frequency Magnitude\n(Log Scale, Shifted)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Plot 3: Frequency Phase
    im3 = axes[2].imshow(phase, cmap='twilight') # Twilight is good for cyclic phase (-pi to pi)
    axes[2].set_title("Frequency Phase\n(-pi to pi)")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Plot 4: Inverse FFT (Sanity Check)
    im4 = axes[3].imshow(reconstructed, cmap='gray')
    axes[3].set_title("Inverse FFT\n(Reconstructed)")
    axes[3].axis('off')
    plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"figures/mnist_sample_visualization_idx{idx}.png", bbox_inches='tight')



transform = T.Compose([
    T.ToTensor(),
])
complex_noise = AddComplexNoise(mean=0.+0.j, std=1.0)

AddRealNoise(0., 0.5),    
T.Lambda(lambda x: torch.clamp(x, 0, 1)),


train_real = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_real  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_ds = ComplexFourierMNIST(train_real)
test_ds  = ComplexFourierMNIST(test_real)

train_complex = ComplexFourierMNIST(train_real, complex_transform=complex_noise)
test_complex = ComplexFourierMNIST(test_real, complex_transform=complex_noise)


train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)


# --- Execute Visualization ---
# Assuming 'train_real' and 'train_ds' are defined from your previous code
print("Visualizing Sample Index 0...")
visualize_single_sample(train_real, train_ds, idx=0)


num_epochs = 50

# activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh", "c_elu", "c_gelu"]
activations_to_test = ["cardioid", "c_relu", "c_elu", "c_gelu"]
# activations_to_test = ["c_elu", "c_gelu"]
learning_rates = [3e-3, 1e-3, 3e-4]
# learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for lr in learning_rates:
    # Run experiment
    lr_history = experiment_lr(lr, num_epochs, activations_to_test, train_loader, test_loader)
    print("\nGenerating Comparison Graphs...")
    plot_comparison(lr_history, num_epochs, str(lr))









