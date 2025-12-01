# %%
import h5py
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
import torchcvnn.nn as c_nn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

# %% [markdown]
# ### Test loading

# %%
# path = "/mnt/i/mridata/brain_multicoil_train_batch_0/multicoil_train/file_brain_AXT2_200_2000057.h5"

activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu"]

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
def rss_combine(img_mc):
    """
    Root-sum-of-squares coil combination.
    img_mc: [num_coils, Ny, Nx], complex
    returns: [Ny, Nx], real
    """
    return torch.sqrt((img_mc.abs() ** 2).sum(dim=0) + 1e-8)

# %%
def simple_mask(Ny, accel=4, device="cpu"):
    mask = torch.zeros(Ny, dtype=torch.float32, device=device)
    mask[::accel] = 1.0
    center = Ny // 2
    mask[center-4:center+4] = 1.0
    return mask

# %%
class FastMRIDataset(Dataset):
    def __init__(self, h5_paths, accel=4, max_slices=None):
        self.paths = h5_paths
        self.accel = accel

        self.samples = []  # list of (path, slice_idx)
        for p in self.paths:
            with h5py.File(p, "r") as f:
                num_slices = f["kspace"].shape[0]
            for s in range(num_slices):
                self.samples.append((p, s))
                if max_slices is not None and len(self.samples) >= max_slices:
                    break
            if max_slices is not None and len(self.samples) >= max_slices:
                break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, s = self.samples[idx]
        with h5py.File(path, "r") as f:
            kspace = f["kspace"][s]  # [num_coils, Ny, Nx], complex64
            kspace = torch.from_numpy(kspace)      # complex64

            # simple 1D undersampling mask along phase-encode (Ny)
            Ny, Nx = kspace.shape[-2], kspace.shape[-1]
            mask = simple_mask(Ny, accel=self.accel).to(kspace.device)
            mask = mask[:, None]                 # [Ny,1], broadcast over Nx
            kspace_und = kspace * mask          # [coils, Ny, Nx]

        # IFFT2 coil-wise: k-space -> image domain
        img_full_mc = torch.fft.ifft2(kspace, norm="ortho")      # [coils,Ny,Nx]
        img_und_mc  = torch.fft.ifft2(kspace_und, norm="ortho")  # [coils,Ny,Nx]

        # Coil combine: RSS magnitude image as target (Y)
        img_full_rss = rss_combine(img_full_mc)   # [Ny,Nx], real

        # For complex model input, you can either:
        # A) use coil-combined complex input (e.g., sum of coils)
        img_und_comb = img_und_mc.sum(dim=0)      # [Ny,Nx], complex

        # add channel dim for Conv2d / complex Conv2d: [1,H,W]
        X_complex = img_und_comb.unsqueeze(0)     # complex [1,H,W]
        Y_target = img_full_rss.unsqueeze(0)      # real [1,H,W]

        return X_complex, Y_target


# %% [markdown]
# ### For single files

# %%
class SingleFastMRIDataset(Dataset):
    def __init__(self, path, accel=4, max_slices=None):
        self.path = path
        self.accel = accel

        # Just count slices in this file
        with h5py.File(self.path, "r") as f:
            num_slices = f["kspace"].shape[0]

        self.slice_indices = list(range(num_slices))
        if max_slices is not None:
            self.slice_indices = self.slice_indices[:max_slices]

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        s = self.slice_indices[idx]
        with h5py.File(self.path, "r") as f:
            kspace = f["kspace"][s]   # [num_coils, Ny, Nx] or [Ny,Nx]
            kspace = torch.from_numpy(kspace)      # complex64

        # ----- simple mask along phase-encode dim -----
        Ny, Nx = kspace.shape[-2], kspace.shape[-1]
        mask = simple_mask(Ny, accel=self.accel).to(kspace.device)
        mask = mask[:, None]           # [Ny,1]
        kspace_und = kspace * mask     # same shape as kspace

        # ----- go to image domain -----
        img_full = torch.fft.ifft2(kspace, norm="ortho")      # complex
        img_und  = torch.fft.ifft2(kspace_und, norm="ortho")  # complex

        # if multicoil: coil combine with RSS
        if img_full.ndim == 3:  # [coils,Ny,Nx]
            img_full_rss = rss_combine(img_full)  # [Ny,Nx], real
            # simple coil-combined complex input (sum of coils)
            img_und_comb = img_und.sum(dim=0)     # [Ny,Nx], complex
        else:  # single-coil [Ny,Nx]
            img_full_rss = img_full.abs()
            img_und_comb = img_und

        X_complex = img_und_comb.unsqueeze(0)   # [1,H,W], complex
        Y_target  = img_full_rss.unsqueeze(0)   # [1,H,W], float

        return X_complex, Y_target


# %%
# train_file = sorted(glob("multicoil_train/file_brain_*.h5"))
# val_file   = sorted(glob("multicoil_val/file_brain_*.h5"))  

# train_ds = FastMRIDataset(train_files, accel=4, max_slices=2000)
# val_ds   = FastMRIDataset(val_files,   accel=4, max_slices=500)


train_path = "/project/def-hamarneh/eastwood/CVNN/multicore_train/file_brain_AXT2_200_2000057.h5"
val_path   = "/project/def-hamarneh/eastwood/CVNN/multicore_val/file_brain_AXT2_200_2000022.h5"

train_ds = SingleFastMRIDataset(train_path, accel=4, max_slices=2000)
val_ds   = SingleFastMRIDataset(val_path,   accel=4, max_slices=500)

cv_train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=4, pin_memory=True)
cv_val_loader   = DataLoader(val_ds,   batch_size=12, shuffle=False, num_workers=4, pin_memory=True)

x0, y0 = next(iter(cv_train_loader))
print("X_complex batch:", x0.shape, x0.dtype)  # [B,1,H,W], complex
print("Y_target batch:", y0.shape, y0.dtype)   # [B,1,H,W], float32


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
class ComplexMRIUNetSmall(nn.Module):
    def __init__(self, act_name="cardioid"):
        super().__init__()
        act = get_complex_activation(act_name)

        # Encoder blocks - Use PyTorch's nn.Conv2d (supports complex!)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(16),
            act,
            nn.Conv2d(16, 16, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(16),
            act,
        )
        self.pool1 = c_nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(32),
            act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(32),
            act,
        )
        self.pool2 = c_nn.AvgPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(64),
            act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(64),
            act,
        )

        # Decoder blocks
        self.up2 = c_nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec2 = nn.Sequential(
            c_nn.BatchNorm2d(32),
            act,
            nn.Conv2d(64, 32, kernel_size=3, padding=1, dtype=torch.complex64),  # 32+32=64
            c_nn.BatchNorm2d(32),
            act,
        )

        self.up1 = c_nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.dec1 = nn.Sequential(
            c_nn.BatchNorm2d(16),
            act,
            nn.Conv2d(32, 16, kernel_size=3, padding=1, dtype=torch.complex64),  # 16+16=32
            c_nn.BatchNorm2d(16),
            act,
        )

        # Final conv to 1 channel
        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1, dtype=torch.complex64)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.out_conv(d1)
        return out

# %%
def visualize_reconstruction_scaled(model, val_loader, num_samples=3, tag="modrelu"):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.to(device)      # [B,1,H,W] complex
        y = y.to(device)      # [B,1,H,W] real

        y_hat = model(x)      # [B,1,H,W] complex

        # Compute a global scaling factor: match mean magnitude
        tgt_mean = y.mean().item()
        pred_mean = y_hat.abs().mean().item()
        eps = 1e-8
        scale = pred_mean / (tgt_mean + eps)
        print(f"Scaling target by factor ≈ {scale:.3e}")

        # Scale target for visualization only
        y_scaled = y * scale

        x_cpu = x.cpu()
        y_cpu = y_scaled.cpu()
        y_hat_cpu = y_hat.cpu()

        def norm_img(img):
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                return (img - vmin) / (vmax - vmin)
            else:
                return np.zeros_like(img)

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            in_mag   = np.fft.fftshift(x_cpu[i,0].abs().numpy())
            tgt_mag  = np.fft.fftshift(y_cpu[i,0].numpy())         # scaled target
            pred_mag = np.fft.fftshift(y_hat_cpu[i,0].abs().numpy())

            print(f"Sample {i}: in[{in_mag.min():.3e},{in_mag.max():.3e}] "
                  f"tgt_scaled[{tgt_mag.min():.3e},{tgt_mag.max():.3e}] "
                  f"pred[{pred_mag.min():.3e},{pred_mag.max():.3e}]")

            in_disp   = norm_img(np.log1p(in_mag))
            tgt_disp  = norm_img(np.log1p(tgt_mag))
            pred_disp = norm_img(np.log1p(pred_mag))

            axes[i,0].imshow(in_disp, cmap="gray")
            axes[i,0].set_title("Input (fftshift, log)")
            axes[i,0].axis("off")

            axes[i,1].imshow(tgt_disp, cmap="gray")
            axes[i,1].set_title("Target×scale (fftshift, log)")
            axes[i,1].axis("off")

            axes[i,2].imshow(pred_disp, cmap="gray")
            axes[i,2].set_title("Prediction (fftshift, log)")
            axes[i,2].axis("off")

        print("orig target mean/std:", y.mean().item(), y.std().item())
        print("pred abs mean/std:", y_hat.abs().mean().item(), y_hat.abs().std().item())

        plt.suptitle(f"Reconstruction – {tag} (scaled target for viz)")
        plt.tight_layout()
        plt.show()


# %%
def calculate_psnr(pred, target, max_val=1.0):
    """Simple PSNR calculation"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

# %%
def mag_mse(pred, target):
    diff = pred.abs() - target
    return (diff ** 2).mean()

# %%
def train_one_epoch(model, loader, optimizer, epoch, tag="cardioid", log_grad_norm=False):
    model.train()
    total_loss, total = 0.0, 0
    grad_norms = []  # per-batch grad norm if logging

    for x, y in tqdm(loader, desc=f"[{tag}] Train {epoch}", leave=False):
        x = x.to(device)   # complex
        y = y.to(device)   # real

        optimizer.zero_grad()
        y_hat = model(x)   # complex
        loss = mag_mse(y_hat, y)
        loss.backward()

        if log_grad_norm:
            with torch.no_grad():
                # global L2 gradient norm
                sq_sum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        sq_sum += p.grad.pow(2).sum().item()
                grad_norm = (sq_sum ** 0.5)
                grad_norms.append(grad_norm)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    avg_loss = total_loss / total
    print(f"[{tag}] Epoch {epoch} | loss={avg_loss:.6f}")

    return avg_loss, grad_norms  # avg_loss per epoch, and list of batch norms


# %%
def evaluate(model, loader, tag="cardioid"):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[{tag}] Eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = mag_mse(y_hat, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    avg_loss = total_loss / total
    print(f"[{tag}] Val | loss={avg_loss:.6f}")
    return avg_loss

# %%
def complex_to_two_channels(x):
    # x: [B,1,H,W] complex
    xr = torch.view_as_real(x)        # [B,1,H,W,2]
    xr = xr.squeeze(1).permute(0, 3, 1, 2)  # [B,2,H,W]
    return xr.float()


# %% [markdown]
# ### LR Sweeping

# %%
def lr_sweep_for_activation(act_name, lrs, num_epochs=10):
    results = {}

    for lr in lrs:
        print(f"\n=== {act_name} | lr={lr} ===")
        model = ComplexMRIUNetSmall(act_name=act_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        grad_norms_epoch = []

        for epoch in range(1, num_epochs + 1):
            train_loss, grad_norms_batch = train_one_epoch(
                model, cv_train_loader, optimizer, epoch,
                tag=f"{act_name}_lr{lr}", log_grad_norm=True
            )
            val_loss = evaluate(model, cv_val_loader, tag=f"{act_name}_lr{lr}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if len(grad_norms_batch) > 0:
                grad_norms_epoch.append(sum(grad_norms_batch) / len(grad_norms_batch))
            else:
                grad_norms_epoch.append(0.0)

        results[lr] = {
            "train_loss": train_losses,
            "val_loss":   val_losses,
            "grad_norm":  grad_norms_epoch,
        }

    return results


# %%
# lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-2]
num_epochs = 60

all_results = {}  # all_results[act_name][lr] = dict with curves

for act_name in activations_to_test:
    print(f"\n##### Sweeping LRs for activation: {act_name} #####")
    all_results[act_name] = lr_sweep_for_activation(act_name, lrs, num_epochs=num_epochs)



# %%
# Save
torch.save(all_results, "cvnn_lr_sweep_results.pt")

# Later, reload
# all_results = torch.load("cvnn_lr_sweep_results.pt")


# %%
import csv

def export_results_to_csv(all_results, path="cvnn_lr_sweep_results.csv"):
    """
    all_results[act_name][lr] = {
        'train_loss': [epoch losses],
        'val_loss':   [epoch losses],
        'grad_norm':  [epoch values],
    }
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["activation", "lr", "epoch", "train_loss", "val_loss", "grad_norm"])

        for act_name, lr_dict in all_results.items():
            for lr, metrics in lr_dict.items():
                train_loss = metrics["train_loss"]
                val_loss   = metrics["val_loss"]
                grad_norm  = metrics["grad_norm"]
                n_epochs = len(train_loss)
                for epoch in range(n_epochs):
                    writer.writerow([
                        act_name,
                        lr,
                        epoch + 1,
                        train_loss[epoch],
                        val_loss[epoch],
                        grad_norm[epoch],
                    ])

# Call after sweeps:
export_results_to_csv(all_results, "cvnn_lr_sweep_results.csv")


# %%
# def plot_val_loss_for_activation(all_results, act_name):
#     plt.figure(figsize=(8,5))
#     for lr, metrics in all_results[act_name].items():
#         plt.plot(metrics["val_loss"], label=f"lr={lr}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Val loss")
#     plt.title(f"Val loss vs epoch – {act_name}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_grad_norm_for_activation(all_results, act_name):
#     plt.figure(figsize=(8,5))
#     for lr, metrics in all_results[act_name].items():
#         plt.plot(metrics["grad_norm"], label=f"lr={lr}")
#     plt.xlabel("Epoch")
#     plt.ylabel("Avg grad L2 norm per epoch")
#     plt.title(f"Grad norm vs epoch – {act_name}")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # %%
# plot_val_loss_for_activation(all_results, "modrelu")
# plot_grad_norm_for_activation(all_results, "modrelu")



# %%
class SingleFastMRIDatasetReal(Dataset):
    def __init__(self, path, accel=4, max_slices=None):
        self.inner = SingleFastMRIDataset(path, accel=accel, max_slices=max_slices)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        X_complex, Y_target = self.inner[idx]      # X_complex: [1,H,W] complex
        # Convert to 2-channel real: [2,H,W]
        xr = torch.view_as_real(X_complex)        # [1,H,W,2]
        xr = xr.squeeze(0).permute(2, 0, 1)       # [2,H,W]
        X_real = xr.float()
        return X_real, Y_target.float()


# %%
train_ds_real = SingleFastMRIDatasetReal(train_path, accel=4, max_slices=2000)
val_ds_real   = SingleFastMRIDatasetReal(val_path,   accel=4, max_slices=500)

rv_train_loader = DataLoader(train_ds_real, batch_size=12, shuffle=True,
                             num_workers=4, pin_memory=True)
rv_val_loader   = DataLoader(val_ds_real,   batch_size=12, shuffle=False,
                             num_workers=4, pin_memory=True)


# %%
class RealMRIUNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        act = nn.ReLU(inplace=True)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            act,
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            act,
        )
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
        )

        # Decoder level 2 (H/4 → H/2)
        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # after concat: 32 (up) + 32 (enc2) = 64
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
        )

        # Decoder level 1 (H/2 → H)
        self.up1 = nn.ConvTranspose2d(
            32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # after concat: 16 (up) + 16 (enc1) = 32
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            act,
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            act,
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)  # [B,64,H/2,W/2]
        d2 = self.dec2(u2)               # [B,32,H/2,W/2]

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)  # [B,32,H,W]
        d1 = self.dec1(u1)               # [B,16,H,W]

        out = self.out_conv(d1)          # [B,1,H,W]
        return out


# %%
def visualize_reconstruction_scaled_real(model, val_loader, num_samples=3, tag="RealUNet"):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.to(device)      # [B,2,H,W] float
        y = y.to(device)      # [B,1,H,W] float

        y_hat = model(x)      # [B,1,H,W] float

        x_cpu = x.cpu()
        y_cpu = y.cpu()
        y_hat_cpu = y_hat.cpu()

        def norm_img(img):
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                return (img - vmin) / (vmax - vmin)
            else:
                return np.zeros_like(img)

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            in_mag   = np.fft.fftshift(np.abs(x_cpu[i,0].numpy()))
            tgt_mag  = np.fft.fftshift(y_cpu[i,0].numpy())
            pred_mag = np.fft.fftshift(y_hat_cpu[i,0].numpy())

            print(f"Sample {i}: in[{in_mag.min():.3e},{in_mag.max():.3e}] "
                  f"tgt[{tgt_mag.min():.3e},{tgt_mag.max():.3e}] "
                  f"pred[{pred_mag.min():.3e},{pred_mag.max():.3e}]")

            # Optional: boost target for visualization only
            tgt_boost = tgt_mag * 1e3  # adjust factor if needed

            in_disp   = norm_img(np.log1p(in_mag))
            tgt_disp  = norm_img(np.log1p(tgt_boost))
            pred_disp = norm_img(np.log1p(pred_mag))

            axes[i,0].imshow(in_disp,   cmap="gray")
            axes[i,0].set_title("Input (fftshift, log)")
            axes[i,0].axis("off")

            axes[i,1].imshow(tgt_disp,  cmap="gray")
            axes[i,1].set_title("Target (scaled, fftshift, log)")
            axes[i,1].axis("off")

            axes[i,2].imshow(pred_disp, cmap="gray")
            axes[i,2].set_title("Prediction (fftshift, log)")
            axes[i,2].axis("off")

        print("orig target mean/std:", y.mean().item(), y.std().item())
        print("pred mean/std:", y_hat.mean().item(), y_hat.std().item())

        plt.suptitle(f"Reconstruction – {tag}")
        plt.tight_layout()
        plt.show()


# %%
def real_mse(pred, target):
    # both [B,1,H,W] real
    # if shapes ever differ (they shouldn't with this U-Net), you can interpolate target
    diff = pred - target
    return (diff ** 2).mean()


# %%
def train_one_epoch_real(model, loader, optimizer, epoch, tag="Real", log_grad_norm=False):
    model.train()
    total_loss, total = 0.0, 0
    grad_norms = []
    for x, y in tqdm(loader, desc=f"[{tag}] Train {epoch}", leave=False):
        x = x.to(device)   # [B,2,H,W] float
        y = y.to(device)   # [B,1,H,W] float

        optimizer.zero_grad()
        y_hat = model(x)   # [B,1,H,W] float
        loss = real_mse(y_hat, y)
        loss.backward()

        if log_grad_norm:
            with torch.no_grad():
                sq_sum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        sq_sum += p.grad.pow(2).sum().item()
                grad_norms.append(sq_sum ** 0.5)


        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    print(f"[{tag}] Epoch {epoch} | loss={total_loss/total:.6f}")


def evaluate_real(model, loader, tag="Real"):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[{tag}] Eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = real_mse(y_hat, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    avg_loss = total_loss / total
    print(f"[{tag}] Val | loss={avg_loss:.6f}")
    return avg_loss


# %%
# real_model = RealMRIUNetSmall().to(device)
# optimizer_real = optim.Adam(real_model.parameters(), lr=1e-3)

# for epoch in range(0, 60):
#     train_one_epoch_real(real_model, rv_train_loader, optimizer_real, epoch, tag="RealUNet")
#     evaluate_real(real_model, rv_val_loader, tag="RealUNet")


# visualize_reconstruction_scaled_real(real_model, rv_val_loader, num_samples=3, tag="RealUNet")

def run_real_unet_experiment(num_epochs=60, lr=1e-3, tag="RealUNet"):
    model = RealMRIUNetSmall().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    grad_norms_epoch = []

    for epoch in range(1, num_epochs + 1):
        train_loss, grad_norms_batch = train_one_epoch_real(
            model, rv_train_loader, optimizer, epoch, tag=tag, log_grad_norm=True
        )
        val_loss = evaluate_real(model, rv_val_loader, tag=tag)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if grad_norms_batch:
            grad_norms_epoch.append(sum(grad_norms_batch) / len(grad_norms_batch))
        else:
            grad_norms_epoch.append(0.0)

    results = {
        "model": model,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "grad_norm": grad_norms_epoch,
        "lr": lr,
    }
    return results

real_results = run_real_unet_experiment(num_epochs=60, lr=1e-3, tag="RealUNet")
# visualize_reconstruction_scaled_real(real_results["model"], rv_val_loader, num_samples=3, tag="RealUNet")

# Optionally save for later comparison
torch.save(real_results, "real_unet_results.pt")


# %%



