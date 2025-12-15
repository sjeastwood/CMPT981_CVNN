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


# %%
# activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu", "c_sigmoid", "c_tanh", "c_elu", "c_gelu"]
activations_to_test = ["modrelu","zrelu", "cardioid", "c_relu"]

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


# %% [markdown]
# ### For bulk dataset in the folder

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
            kspace = f["kspace"][s]           # [num_coils, Ny, Nx], complex64
            kspace = torch.from_numpy(kspace)  # complex64

            Ny, Nx = kspace.shape[-2], kspace.shape[-1]
            mask = simple_mask(Ny, accel=self.accel).to(kspace.device)
            mask = mask[:, None]             # [Ny,1]
            kspace_und = kspace * mask       # [coils,Ny,Nx]

        # IFFT2 coil-wise: k-space -> image domain
        img_full_mc = torch.fft.ifft2(kspace, norm="ortho")      # [coils,Ny,Nx] complex
        img_und_mc  = torch.fft.ifft2(kspace_und, norm="ortho")  # [coils,Ny,Nx] complex

        # Coil-combined complex ground truth and input (simple sum over coils)
        img_full_comb = img_full_mc.sum(dim=0)   # [Ny,Nx] complex
        img_und_comb  = img_und_mc.sum(dim=0)    # [Ny,Nx] complex

        X_complex = img_und_comb.unsqueeze(0)    # [1,H,W] complex
        Y_complex = img_full_comb.unsqueeze(0)   # [1,H,W] complex

        return X_complex, Y_complex



# %% [markdown]
# ### For single files

# %%
class SingleFastMRIDataset(Dataset):
    def __init__(self, path, accel=4, max_slices=None):
        self.path = path
        self.accel = accel

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
            kspace = f["kspace"][s]          # [num_coils, Ny, Nx] or [Ny,Nx]
            kspace = torch.from_numpy(kspace)  # complex64

        Ny, Nx = kspace.shape[-2], kspace.shape[-1]
        mask = simple_mask(Ny, accel=self.accel).to(kspace.device)
        mask = mask[:, None]                # [Ny,1]
        kspace_und = kspace * mask

        # image domain
        img_full = torch.fft.ifft2(kspace, norm="ortho")      # complex
        img_und  = torch.fft.ifft2(kspace_und, norm="ortho")  # complex

        if img_full.ndim == 3:   # [coils,Ny,Nx]
            # coil-combined complex ground truth (simple sum here)
            img_full_comb = img_full.sum(dim=0)   # [Ny,Nx] complex
            img_und_comb  = img_und.sum(dim=0)    # [Ny,Nx] complex
        else:                    # [Ny,Nx]
            img_full_comb = img_full
            img_und_comb  = img_und

        X_complex = img_und_comb.unsqueeze(0)   # [1,H,W] complex
        Y_complex = img_full_comb.unsqueeze(0)  # [1,H,W] complex

        return X_complex, Y_complex


# %%
# train_file = sorted(glob("multicoil_train/file_brain_*.h5"))
# val_file   = sorted(glob("multicoil_val/file_brain_*.h5"))  

train_path = "/project/def-hamarneh/eastwood/CVNN/multicore_train/"
val_path   = "/project/def-hamarneh/eastwood/CVNN/multicore_val/"

train_files = sorted(glob.glob(train_path + "*.h5"))
val_files   = sorted(glob.glob(val_path + "*.h5"))

train_ds = FastMRIDataset(train_files, accel=4, max_slices=2000)
val_ds   = FastMRIDataset(val_files,   accel=4, max_slices=500)


# train_path = "/project/def-hamarneh/eastwood/CVNN/multicore_train/file_brain_AXT2_200_2000057.h5"
# val_path   = "/project/def-hamarneh/eastwood/CVNN/multicore_val/file_brain_AXT2_200_2000022.h5"

# train_ds = SingleFastMRIDataset(train_path, accel=4, max_slices=2000)
# val_ds   = SingleFastMRIDataset(val_path,   accel=4, max_slices=500)


cv_train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
cv_val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

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
def complex_mse(pred, target):

    diff = pred - target
    return (diff.real**2 + diff.imag**2).mean()



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

        # Initialize weights
        self._init_complex_weights()

    def _init_complex_weights(self):
        """
        Initialize complex-valued weights using Kaiming/He initialization.
        For complex weights, we initialize real and imaginary parts independently
        with variance/2 for each part to maintain the overall variance.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # if m.weight.is_complex():
                #     # Get weight as view_as_real: [..., 2] where last dim is [real, imag]
                #     weight_real_view = torch.view_as_real(m.weight.data)

                #     # Calculate fan_in for Kaiming initialization
                #     fan_in = m.weight.size(1) * m.kernel_size[0] * m.kernel_size[1]

                #     # Kaiming uniform bound with variance split between real and imag
                #     # For complex: variance should be split, so each part gets sqrt(1/fan_in)
                #     bound = (1.0 / fan_in) ** 0.5

                #     # Initialize real and imaginary parts independently
                #     nn.init.uniform_(weight_real_view[..., 0], -bound, bound)  # real part
                #     nn.init.uniform_(weight_real_view[..., 1], -bound, bound)  # imaginary part

                #     # Update the actual complex weight
                #     m.weight.data = torch.view_as_complex(weight_real_view.contiguous())

                # # Initialize bias if present
                # if m.bias is not None and m.bias.is_complex():
                #     bias_real_view = torch.view_as_real(m.bias.data)
                #     bound = (1.0 / fan_in) ** 0.5
                #     nn.init.uniform_(bias_real_view[..., 0], -bound, bound)
                #     nn.init.uniform_(bias_real_view[..., 1], -bound, bound)
                #     m.bias.data = torch.view_as_complex(bias_real_view.contiguous())

                #----- TORCHCVNN KAIMING INITIALIZATION -----
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, c_nn.ConvTranspose2d)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        c_nn.init.complex_kaiming_uniform_(m.weight, mode="fan_in")
                    if hasattr(m, 'bias') and m.bias is not None and m.bias.is_complex():
                        nn.init.uniform_(m.bias.real, -0.01, 0.01)
                        nn.init.uniform_(m.bias.imag, -0.01, 0.01)


                #----- TORCHCVNN XAVIER INITIALIZATION -----
                # if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, c_nn.ConvTranspose2d)):
                #     if hasattr(m, 'weight') and m.weight is not None:
                #         c_nn.init.complex_xavier_uniform_(m.weight)
                #     if hasattr(m, 'bias') and m.bias is not None and m.bias.is_complex():
                #         nn.init.uniform_(m.bias.real, -0.01, 0.01)
                #         nn.init.uniform_(m.bias.imag, -0.01, 0.01)

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
def clip_and_norm_log_mag(mag, clip_percent=99.9):
    # log10 magnitude
    logmag = np.log10(mag + 1e-12)
    # clip high end to a percentile to avoid one super-bright DC dominating
    hi = np.percentile(logmag, clip_percent)
    lo = np.percentile(logmag, 1.0)
    logmag = np.clip(logmag, lo, hi)
    vmin, vmax = logmag.min(), logmag.max()
    if vmax > vmin:
        return (logmag - vmin) / (vmax - vmin)
    else:
        return np.zeros_like(logmag)


# %%
def visualize_reconstruction_scaled(model, val_loader, num_samples=3, tag="modrelu"):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.to(device)        # [B,1,H,W] complex
        y = y.to(device)        # [B,1,H,W] complex

        y_hat = model(x)        # [B,1,H,W] complex

        # Magnitudes for scaling + display
        y_mag     = y.abs()
        y_hat_mag = y_hat.abs()

        tgt_mean  = y_mag.mean().item()
        pred_mean = y_hat_mag.mean().item()
        eps = 1e-8
        scale = pred_mean / (tgt_mean + eps)
        print(f"Scaling target magnitude by factor ≈ {scale:.3e}")

        y_mag_scaled = y_mag * scale  # [B,1,H,W] real

        x_cpu     = x.cpu()
        y_mag_cpu = y_mag_scaled.cpu()
        y_hat_cpu = y_hat_mag.cpu()

        def norm_img(img):
            vmin, vmax = img.min(), img.max()
            if vmax > vmin:
                return (img - vmin) / (vmax - vmin)
            else:
                return np.zeros_like(img)

        # ----- Image-space visualization -----
        fig_img, axes_img = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes_img = axes_img.reshape(1, -1)

        for i in range(num_samples):
            in_mag   = np.fft.fftshift(x_cpu[i,0].abs().numpy())
            tgt_mag  = np.fft.fftshift(y_mag_cpu[i,0].numpy())
            pred_mag = np.fft.fftshift(y_hat_cpu[i,0].numpy())

            print(f"[Image] Sample {i}: in[{in_mag.min():.3e},{in_mag.max():.3e}] "
                  f"tgt_scaled[{tgt_mag.min():.3e},{tgt_mag.max():.3e}] "
                  f"pred[{pred_mag.min():.3e},{pred_mag.max():.3e}]")

            in_disp   = norm_img(np.log1p(in_mag))
            tgt_disp  = norm_img(np.log1p(tgt_mag))
            pred_disp = norm_img(np.log1p(pred_mag))

            axes_img[i,0].imshow(in_disp,  cmap="gray")
            axes_img[i,0].set_title("Input (img, fftshift, log)")
            axes_img[i,0].axis("off")

            axes_img[i,1].imshow(tgt_disp, cmap="gray")
            axes_img[i,1].set_title("Target×scale (img, fftshift, log)")
            axes_img[i,1].axis("off")

            axes_img[i,2].imshow(pred_disp, cmap="gray_r")
            axes_img[i,2].set_title("Prediction (img, fftshift, log)")
            axes_img[i,2].axis("off")

        print("orig target mag mean/std:", y_mag.mean().item(), y_mag.std().item())
        print("pred mag mean/std:", y_hat_mag.mean().item(), y_hat_mag.std().item())

        fig_img.suptitle(f"Reconstruction – {tag} (Image space)")
        fig_img.tight_layout()
        plt.show()

        # ----- Fourier-space visualization -----
        fig_k, axes_k = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes_k = axes_k.reshape(1, -1)

        for i in range(num_samples):
            # Compute FFT of image-domain complex data
            xin  = x_cpu[i,0].numpy()
            ytgt = y[i,0].cpu().numpy()        # original complex target (unscaled)
            yph  = y_hat[i,0].cpu().numpy()

            k_in   = np.fft.fftshift(np.fft.fft2(xin, norm="ortho"))
            k_tgt  = np.fft.fftshift(np.fft.fft2(ytgt, norm="ortho"))
            k_pred = np.fft.fftshift(np.fft.fft2(yph, norm="ortho"))

            k_in_mag   = np.abs(k_in)
            k_tgt_mag  = np.abs(k_tgt)
            k_pred_mag = np.abs(k_pred)

            print(f"[K-space] Sample {i}: in[{k_in_mag.min():.3e},{k_in_mag.max():.3e}] "
                  f"tgt[{k_tgt_mag.min():.3e},{k_tgt_mag.max():.3e}] "
                  f"pred[{k_pred_mag.min():.3e},{k_pred_mag.max():.3e}]")

            kin_disp   = clip_and_norm_log_mag(k_in_mag)
            ktgt_disp  = clip_and_norm_log_mag(k_tgt_mag)
            kpred_disp = clip_and_norm_log_mag(k_pred_mag)
            
            axes_k[i,0].imshow(kin_disp,   cmap="gray")
            axes_k[i,0].set_title("Input (k-space, fftshift, log)")
            axes_k[i,0].axis("off")

            axes_k[i,1].imshow(ktgt_disp,  cmap="gray")
            axes_k[i,1].set_title("Target (k-space, fftshift, log)")
            axes_k[i,1].axis("off")

            axes_k[i,2].imshow(kpred_disp, cmap="gray")
            axes_k[i,2].set_title("Prediction (k-space, fftshift, log)")
            axes_k[i,2].axis("off")

        fig_k.suptitle(f"Reconstruction – {tag} (K-space)")
        fig_k.tight_layout()
        plt.show()


# %%
def calculate_psnr(pred, target, max_val=1.0):
    """Simple PSNR calculation"""
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

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
        loss = complex_mse(y_hat, y)
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
            loss = complex_mse(y_hat, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    avg_loss = total_loss / total
    print(f"[{tag}] Val | loss={avg_loss:.6f}")
    return avg_loss

# %% [markdown]
# ### LR Sweeping

# %%
def lr_sweep_for_activation(act_name, lrs, num_epochs=60):
    results = {}

    

    for lr in lrs:
        print(f"\n=== {act_name} | lr={lr} ===")
        model = ComplexMRIUNetSmall(act_name=act_name).to(device)
        # opt = torch.optim.SGD(model.parameters(), lr=lr)
        # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        # opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True) 

        train_losses = []
        val_losses = []
        grad_norms_epoch = []

        for epoch in range(1, num_epochs + 1):
            train_loss, grad_norms_batch = train_one_epoch(
                model, cv_train_loader, opt, epoch,
                tag=f"{act_name}_lr{lr}", log_grad_norm=True
            )
            val_loss = evaluate(model, cv_val_loader, tag=f"{act_name}_lr{lr}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if len(grad_norms_batch) > 0:
                grad_norms_epoch.append(sum(grad_norms_batch) / len(grad_norms_batch))
            else:
                grad_norms_epoch.append(0.0)

        results = {
            "train_loss": train_losses,
            "val_loss":   val_losses,
            "grad_norm":  grad_norms_epoch,
        }

    return results


# %%
# lrs = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
# lrs = [1e-2, 1e-3, 1e-4, 1e-5]
lrs = [1e-2]
num_epochs = 60

all_results = {}  

for act_name in activations_to_test:
    print(f"\n##### Sweeping LRs for activation: {act_name} #####")
    all_results[act_name] = lr_sweep_for_activation(act_name, lrs, num_epochs=num_epochs)



# %%
# Save
torch.save(all_results, "cvnn_opt_sweep_results_adam_amsgrad.pt")

# Reload
# all_results = torch.load("cvnn_lr_sweep_results.pt")


# %% [markdown]
# # REAL NETWORK

# %%
class SingleFastMRIDatasetReal(Dataset):
    def __init__(self, path, accel=4, max_slices=None):
        self.inner = SingleFastMRIDataset(path, accel=accel, max_slices=max_slices)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        X_complex, Y_complex = self.inner[idx]    # both [1,H,W] complex

        # Input: 2‑channel real [2,H,W]
        Xr = torch.view_as_real(X_complex)        # [1,H,W,2]
        Xr = Xr.squeeze(0).permute(2, 0, 1)       # [2,H,W]

        # Target: 2‑channel real [2,H,W]
        Yr = torch.view_as_real(Y_complex)        # [1,H,W,2]
        Yr = Yr.squeeze(0).permute(2, 0, 1)       # [2,H,W]

        return Xr.float(), Yr.float()


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
        self.pool1 = nn.AvgPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
        )
        self.pool2 = nn.AvgPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            act,
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            act,
        )

        self.out_conv = nn.Conv2d(16, 2, kernel_size=3, padding=1)  # 2 channels: Re, Im

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        out = self.out_conv(d1)   # [B,2,H,W]
        return out


# %%
def visualize_reconstruction_scaled_real(model, val_loader, num_samples=3, tag="RealUNet"):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(val_loader))
        x = x.to(device)      # [B,2,H,W] float  (Re, Im)
        y = y.to(device)      # [B,2,H,W] float  (Re, Im)

        y_hat = model(x)      # [B,2,H,W] float

        x_cpu     = x.cpu()
        y_cpu     = y.cpu()
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
            # Reconstruct complex images from 2 channels
            x_complex     = x_cpu[i,0].numpy() + 1j * x_cpu[i,1].numpy()
            y_complex     = y_cpu[i,0].numpy() + 1j * y_cpu[i,1].numpy()
            y_hat_complex = y_hat_cpu[i,0].numpy() + 1j * y_hat_cpu[i,1].numpy()

            in_mag   = np.fft.fftshift(np.abs(x_complex))
            tgt_mag  = np.fft.fftshift(np.abs(y_complex))
            pred_mag = np.fft.fftshift(np.abs(y_hat_complex))

            print(f"Sample {i}: in[{in_mag.min():.3e},{in_mag.max():.3e}] "
                  f"tgt[{tgt_mag.min():.3e},{tgt_mag.max():.3e}] "
                  f"pred[{pred_mag.min():.3e},{pred_mag.max():.3e}]")

            # Optional: boost target magnitude for visualization only
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

        # Stats on complex magnitudes
        y_mag     = torch.sqrt(y[:,0]**2 + y[:,1]**2)
        y_hat_mag = torch.sqrt(y_hat[:,0]**2 + y_hat[:,1]**2)
        print("orig target mag mean/std:", y_mag.mean().item(), y_mag.std().item())
        print("pred mag mean/std:", y_hat_mag.mean().item(), y_hat_mag.std().item())

        plt.suptitle(f"Reconstruction – {tag}")
        plt.tight_layout()
        plt.show()


# %%
def real_complex_mse(pred, target):

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
        loss = real_complex_mse(y_hat, y)
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

    avg_loss = total_loss / total
    print(f"[{tag}] Epoch {epoch} | loss={avg_loss:.6f}")

    return avg_loss, grad_norms



def evaluate_real(model, loader, tag="Real"):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"[{tag}] Eval", leave=False):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = real_complex_mse(y_hat, y)
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
    avg_loss = total_loss / total
    print(f"[{tag}] Val | loss={avg_loss:.6f}")
    return avg_loss

# %%
def run_real_unet_experiment(num_epochs=60, lr=1e-3, tag="RealUNet"):
    model = RealMRIUNetSmall().to(device)
    # opt = torch.optim.SGD(model.parameters(), lr=lr)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    # opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True) 

    train_losses = []
    val_losses = []
    grad_norms_epoch = []

    for epoch in range(1, num_epochs + 1):
        train_loss, grad_norms_batch = train_one_epoch_real(
            model, rv_train_loader, opt, epoch, tag=tag, log_grad_norm=True
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
    }
    return results

# # %%
real_results = {}

lr = 1e-2
real_results = run_real_unet_experiment(num_epochs=60, lr=lr, tag="RealUNet")

# # %%
torch.save(real_results, f"real_unet_results_adam_amsgrad.pt") 
