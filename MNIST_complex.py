"""
CVNN_MNIST_complex.py

This module implements a comparative experiment between Real-valued and Complex-valued 
Convolutional Neural Networks (CVNN) on a noisy MNIST dataset. 

It includes:
1. Custom Dataset wrapper for introducing spatial or frequency noise.
2. Model definitions for Real and Complex CNNs.
3. Training and Evaluation loops.
4. Experiment runners and logging.
5. Visualization tools for data and training curves.
"""

import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import torchvision
import torchvision.transforms as T

# Assume torchcvnn is installed in the environment
import torchcvnn.nn as c_nn

# --- Configuration & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HISTORY_DIR = "history/complex/std2"

# ==================================================================================================
# 1. UTILITIES
# ==================================================================================================

def get_complex_activation(name: str):
    """Factory for complex activation functions."""
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

def get_optimizer(parameters, name, lr):
    """Factory for optimizers."""
    name = name.lower()
    if name == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif name == 'sgd':
        return optim.SGD(parameters, lr=lr, nesterov=False)
    elif name == 'nesterov':
        return optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Optimizer {name} not supported")

def compute_grad_norm(model):
    """Computes the L2 norm of the model's gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def save_history_to_csv(history, folder, filename):
    """Saves a dictionary of metrics to a CSV file."""
    df = pd.DataFrame(history)
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)

# ==================================================================================================
# 2. DATASET & DATALOADERS
# ==================================================================================================

class UnifiedNoisyMNIST(Dataset):
    """
    MNIST wrapper that adds noise in either the Spatial or Frequency domain 
    and returns complex-valued tensors.
    """
    def __init__(self, base_dataset, mode='real', noise_std=0.1):
        self.base_dataset = base_dataset
        self.noise_mode = mode.lower()
        self.noise_std = noise_std
        self.to_tensor = T.ToTensor()
        
        if self.noise_mode == 'complex':
            self.complex_mean = torch.tensor([0.0, 0.0])
            var = self.noise_std ** 2
            self.complex_cov = torch.tensor([[var, 0.0], [0.0, var]])

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_pil, label = self.base_dataset[idx]
        img_tensor = self.to_tensor(img_pil)

        if self.noise_mode == 'real':
            out_tensor = self._process_spatial_noise(img_tensor)
        elif self.noise_mode == 'complex':
            out_tensor = self._process_freq_noise(img_tensor)
        else:
            raise ValueError(f"Unknown noise_mode: {self.noise_mode}")

        return out_tensor, label

    def _process_spatial_noise(self, img):
        noise = torch.randn_like(img) * self.noise_std
        noisy_img = torch.clamp(img + noise, 0.0, 1.0)
        img_freq = torch.fft.fft2(noisy_img)   
        return img_freq.to(torch.complex64)

    def _process_freq_noise(self, img):
        img_freq = torch.fft.fft2(img) 
        img_view = torch.view_as_real(img_freq)

        dist = MultivariateNormal(loc=self.complex_mean, covariance_matrix=self.complex_cov)
        noise = dist.sample(sample_shape=img_view.shape[:-1])
        noisy_freq = torch.view_as_complex(img_view + noise)
        
        spatial_recon = torch.fft.ifft2(noisy_freq).real
        spatial_clamped = torch.clamp(spatial_recon, 0.0, 1.0)
        final_freq = torch.fft.fft2(spatial_clamped)
        
        return final_freq.to(torch.complex64)

def get_dataloaders(batch_size=64, noise_std=0.2, data_root='./data'):
    print("Loading base MNIST data...")
    raw_train = torchvision.datasets.MNIST(root=data_root, train=True, download=True)
    raw_test = torchvision.datasets.MNIST(root=data_root, train=False, download=True)
    
    # Wrappers
    ds_real_train = UnifiedNoisyMNIST(raw_train, mode='real', noise_std=noise_std)
    ds_real_test  = UnifiedNoisyMNIST(raw_test,  mode='real', noise_std=noise_std)
    ds_complex_train = UnifiedNoisyMNIST(raw_train, mode='complex', noise_std=noise_std)
    ds_complex_test  = UnifiedNoisyMNIST(raw_test,  mode='complex', noise_std=noise_std)

    loaders = {
        'raw_train':  DataLoader(raw_train, batch_size=batch_size, num_workers=4, shuffle=True),
        'raw_test':   DataLoader(raw_test,  batch_size=batch_size, num_workers=4, shuffle=False),
        'real_train': DataLoader(ds_real_train, batch_size=batch_size, num_workers=4, shuffle=True),
        'real_test':  DataLoader(ds_real_test,  batch_size=batch_size, num_workers=4, shuffle=False),
        'comp_train': DataLoader(ds_complex_train, batch_size=batch_size, num_workers=4, shuffle=True),
        'comp_test':  DataLoader(ds_complex_test,  batch_size=batch_size, num_workers=4, shuffle=False),
    }
    return loaders

# ==================================================================================================
# 3. MODELS
# ==================================================================================================

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
        return x.abs() # Magnitude for classification

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

# ==================================================================================================
# 4. TRAINING & EVALUATION
# ==================================================================================================

def train_cvnn_epoch(model, loader, optimizer, criterion, epoch, act_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_norm = 0.0

    pbar = tqdm(loader, desc=f"[{act_name}] Train {epoch}", leave=False)
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = compute_grad_norm(model)
        total_grad_norm += grad_norm
        
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix(loss=total_loss/total, acc=correct/total)

    avg_loss = total_loss / total
    acc = correct / total
    avg_grad_norm = total_grad_norm / len(loader)
    
    print(f"[{act_name}] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Grad: {avg_grad_norm:.4f}")
    return avg_loss, acc, avg_grad_norm

def evaluate_cvnn(model, loader, criterion, act_name):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{act_name}] Eval", leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[{act_name}] Val     | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    return avg_loss, acc

def train_real_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    total_grad_norm = 0.0

    pbar = tqdm(loader, desc=f"[Real CNN] Train {epoch}", leave=False)
    for x, y in pbar:
        y = y.to(DEVICE)
        # Convert Complex Input [B, 1, 28, 28] -> Real Two-Channel [B, 2, 28, 28]
        xr = torch.view_as_real(x)        # [B, 1, 28, 28, 2]
        xr = xr.squeeze(1).permute(0, 3, 1, 2)
        x_input = xr.float().to(DEVICE)

        optimizer.zero_grad()
        logits = model(x_input)
        loss = criterion(logits, y)
        loss.backward()

        grad_norm = compute_grad_norm(model)
        total_grad_norm += grad_norm

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        pbar.set_postfix(loss=total_loss/total, acc=correct/total)

    avg_loss = total_loss / total
    acc = correct / total
    avg_grad_norm = total_grad_norm / len(loader)

    print(f"[Real CNN] Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Grad: {avg_grad_norm:.4f}")
    return avg_loss, acc, avg_grad_norm

def evaluate_real(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[Real CNN] Eval", leave=False)
        for x, y in pbar:
            y = y.to(DEVICE)
            xr = torch.view_as_real(x)        
            xr = xr.squeeze(1).permute(0, 3, 1, 2)
            x_input = xr.float().to(DEVICE)

            logits = model(x_input)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            _, preds = logits.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    print(f"[Real CNN] Val     | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")
    return avg_loss, acc

# ==================================================================================================
# 5. EXPERIMENT RUNNER
# ==================================================================================================

def run_comprehensive_experiment(train_loader, test_loader, activations, lr_list, optimizers, num_epochs=10):
    """
    Runs grid search over optimizers, learning rates, and activation functions.
    Saves logs to CSV.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    print(f"Starting Experiment. Logs will be saved to '{HISTORY_DIR}/'")
    print(f"Configs: {len(optimizers)} Opts x {len(lr_list)} LRs x ({len(activations)} CVNNs + 1 RVNN)")

    criterion = nn.CrossEntropyLoss()

    for opt_name in optimizers:
        for lr in lr_list:
            log_lr = int(-np.log10(lr))
            print(f"\n{'='*40}")
            print(f"Running: Optimizer={opt_name} | LR={lr} (1e-{log_lr})")
            print(f"{'='*40}")

            # --- 1. Real Baseline ---
            model_name = f"RealCNN_{opt_name}_lr{log_lr}"
            print(f"--- Training {model_name} ---")
            
            real_model = RealMnistCNN(use_two_channels=True).to(DEVICE)
            optimizer = get_optimizer(real_model.parameters(), opt_name, lr)
            
            history = {
                "epoch": [], "train_loss": [], "train_acc": [], 
                "val_loss": [], "val_acc": [], "grad_norm": [], "time": []
            }

            for epoch in range(1, num_epochs + 1):
                start = time.time()
                t_loss, t_acc, g_norm = train_real_epoch(real_model, train_loader, optimizer, criterion, epoch)
                v_loss, v_acc = evaluate_real(real_model, test_loader, criterion)
                elapsed = time.time() - start

                history["epoch"].append(epoch)
                history["train_loss"].append(t_loss)
                history["train_acc"].append(t_acc)
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)
                history["grad_norm"].append(g_norm)
                history["time"].append(elapsed)

            save_history_to_csv(history, HISTORY_DIR, f"{model_name}.csv")

            # --- 2. Complex Models (Iterate Activations) ---
            for act in activations:
                model_name = f"CVNN_{act}_{opt_name}_lr{log_lr}"
                print(f"--- Training {model_name} ---")

                model = ComplexMnistCNN(act_name=act).to(DEVICE)
                optimizer = get_optimizer(model.parameters(), opt_name, lr)

                history = {
                    "epoch": [], "train_loss": [], "train_acc": [], 
                    "val_loss": [], "val_acc": [], "grad_norm": [], "time": []
                }

                for epoch in range(1, num_epochs + 1):
                    start = time.time()
                    t_loss, t_acc, g_norm = train_cvnn_epoch(model, train_loader, optimizer, criterion, epoch, act_name=act)
                    v_loss, v_acc = evaluate_cvnn(model, test_loader, criterion, act_name=act)
                    elapsed = time.time() - start

                    history["epoch"].append(epoch)
                    history["train_loss"].append(t_loss)
                    history["train_acc"].append(t_acc)
                    history["val_loss"].append(v_loss)
                    history["val_acc"].append(v_acc)
                    history["grad_norm"].append(g_norm)
                    history["time"].append(elapsed)

                save_history_to_csv(history, HISTORY_DIR, f"{model_name}.csv")

    print("\nAll experiments completed.")

# ==================================================================================================
# 6. VISUALIZATION & ANALYSIS
# ==================================================================================================

def visualize_single_sample(real_ds, complex_ds, idx=0, noise='real'):
    """
    Visualizes the spatial real image, frequency magnitude/phase, and reconstructed image.
    Robustly handles both PIL Images (Raw MNIST) and Tensors (Transformed MNIST).
    """
    real_img, real_label = real_ds[idx]       
    comp_img, comp_label = complex_ds[idx]    

    if isinstance(real_img, torch.Tensor):
        real_plot = real_img.squeeze().numpy()
    else:
        real_plot = np.array(real_img)
        if real_plot.max() > 1.0:
            real_plot = real_plot / 255.0
    
    comp_shifted = torch.fft.fftshift(comp_img.squeeze())
    magnitude = comp_shifted.abs()
    log_magnitude = torch.log(1 + magnitude).numpy()
    phase = comp_shifted.angle().numpy()
    reconstructed = torch.fft.ifft2(comp_img).real.squeeze().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(real_plot, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Real Dataset (Spatial)\nLabel: {real_label}")
    
    axes[1].imshow(log_magnitude, cmap='inferno')
    axes[1].set_title("Complex DS Magnitude\n(Log Scale, Shifted)")
    
    axes[2].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi) 
    axes[2].set_title("Complex DS Phase\n(-pi to pi)")
    
    axes[3].imshow(reconstructed, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title("Complex DS Reconstructed\n(IFFT of Noisy Input)")

    for ax in axes: ax.axis('off')
    
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figures/mnist_sample_visualization_{noise}_noise_idx{idx}.png", bbox_inches='tight')
    print(f"Sample visualization saved to figures/mnist_sample_visualization_{noise}_noise_idx{idx}.png")
    plt.close()
    
def parse_filename(filename):
    """
    Robust regex parser for log filenames.
    Handles:
    - CVNN_{activation}_{opt}_lr{int_log_lr}.csv (e.g. CVNN_c_relu_adam_lr2.csv)
    - RealCNN_{opt}_lr{int_log_lr}.csv           (e.g. RealCNN_adam_lr2.csv)
    """
    cvnn_pattern = r"CVNN_(.+)_([^_]+)_lr(\d+)\.csv"
    real_pattern = r"RealCNN_([^_]+)_lr(\d+)\.csv"
    
    match_cvnn = re.match(cvnn_pattern, filename)
    if match_cvnn:
        act, opt, log_lr = match_cvnn.groups()
        lr_val = 10**(-int(log_lr))
        return {
            "model_type": "CVNN",
            "activation": act,
            "optimizer": opt,
            "learning_rate": float(f"{lr_val:.6g}"),
            "log_lr": int(log_lr),
            "filename": filename
        }

    match_real = re.match(real_pattern, filename)
    if match_real:
        opt, log_lr = match_real.groups()
        lr_val = 10**(-int(log_lr))
        return {
            "model_type": "RealCNN",
            "activation": "Real (Baseline)", 
            "optimizer": opt,
            "learning_rate": float(f"{lr_val:.6g}"),
            "log_lr": int(log_lr),
            "filename": filename
        }       
    return None

def load_logs(log_dir):
    """Loads all CSV logs from directory into a single DataFrame."""
    all_data = []
    if not os.path.exists(log_dir):
        print(f"Directory {log_dir} does not exist.")
        return pd.DataFrame()

    files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
    print(f"Found {len(files)} log files.")
    
    for f in files:
        params = parse_filename(f)
        if params:
            filepath = os.path.join(log_dir, f)
            try:
                df = pd.read_csv(filepath)
                for k, v in params.items():
                    df[k] = v
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
    
    if not all_data:
        print("No valid data found.")
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

def plot_experiment(log_dir, metric, compare_by, fixed_params=None, semilogy=False, output_file=None):
    """
    Generates comparison plots from logs.
    """
    df = load_logs(log_dir)
    if df.empty:
        print("No data found.")
        return

    # Filter
    if fixed_params:
        for key, value in fixed_params.items():
            if key in df.columns:
                if isinstance(value, float):
                    df = df[np.isclose(df[key], value)]
                else:
                    df = df[df[key] == value]
            else:
                print(f"Warning: '{key}' not in columns. Ignoring filter.")
    
    if df.empty:
        print(f"No data matches filter: {fixed_params}")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    x_axis = 'epoch' if 'epoch' in df.columns else df.columns[0]
    
    ax = sns.lineplot(
        data=df, 
        x=x_axis, 
        y=metric, 
        hue=compare_by, 
        style=compare_by, 
        markers=True, 
        dashes=False,
        palette='tab10'
    )
    
    if semilogy:
        ax.set_yscale('log')
        plt.ylabel(f"{metric} (log scale)")
    else:
        plt.ylabel(metric)
        
    title = f"{metric} vs {x_axis}\nComparing: {compare_by}"
    if fixed_params:
        filters_str = ", ".join([f"{k}={v}" for k,v in fixed_params.items()])
        title += f"\n(Fixed: {filters_str})"
    
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5, which="both" if semilogy else "major")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()

# ==================================================================================================
# 7. MAIN EXECUTION
# ==================================================================================================

if __name__ == "__main__":
    # --- Config ---
    BATCH_SIZE = 128
    NOISE_STD = 0.2
    NUM_EPOCHS = 50
    
    # Activation functions to test (including underscores)
    ACTIVATIONS = ["modrelu", "cardioid", "c_relu", "c_elu", "c_gelu"]
    
    # Learning rates (10^-1 to 10^-5)
    LEARNING_RATES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    OPTIMIZERS = ['Adam', 'SGD', 'Nesterov']

    # --- 1. Prepare Data ---
    loaders = get_dataloaders(batch_size=BATCH_SIZE, noise_std=NOISE_STD) 
    
    # --- 2. Visualization Check ---
    # Visualizes the first sample of the training set
    raw_train_ds = loaders['raw_train'].dataset
    real_train_ds = loaders['real_train'].dataset
    comp_train_ds = loaders['comp_train'].dataset
    visualize_single_sample(raw_train_ds, real_train_ds, idx=0, noise='real')
    visualize_single_sample(raw_train_ds, comp_train_ds, idx=0, noise='complex')

    # --- 3. Run Experiments ---
    # Uncomment the line below to run the training loop
    run_comprehensive_experiment(
        loaders['comp_train'], 
        loaders['comp_test'], 
        ACTIVATIONS, 
        LEARNING_RATES, 
        OPTIMIZERS, 
        NUM_EPOCHS
    )

    # --- 4. Plot Results (Example) ---
    # plot_experiment(
    #     log_dir=HISTORY_DIR,
    #     metric="train_loss",
    #     compare_by="activation",
    #     fixed_params={"learning_rate": 0.01, "optimizer": "adam"},
    #     semilogy=True,
    #     output_file="loss_log_scale.png"
    # )