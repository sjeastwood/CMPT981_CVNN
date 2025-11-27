import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchcvnn.nn as c_nn  # complex layers / activations
# If torchcvnn provides a complex MNIST dataset class in its examples:
# from torchcvnn.datasets import MNIST_Fourier  # example name; check examples repo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example complex activation selector using torchcvnn activations.
# Check torchcvnn.nn for names like ModReLU, zReLU, Cardioid, etc.
def get_complex_activation(name: str):
    name = name.lower()
    if name == "modrelu":
        return c_nn.ModReLU()
    elif name == "zrelu":
        return c_nn.ZReLU()
    elif name == "cardioid":
        return c_nn.Cardioid()
    elif name == "c_relu":
        return c_nn.CReLU()
    # Fallback: identity
    return nn.Identity()

class ComplexMnistCNN(nn.Module):
    def __init__(self, act_name="modrelu"):
        super().__init__()
        act = get_complex_activation(act_name)

        self.features = nn.Sequential(
            c_nn.Conv2d(1, 16, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(16),
            act,
            c_nn.Conv2d(16, 32, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(32),
            act,
            c_nn.AvgPool2d(2),  # torchcvnn reimplements some pooling for complex tensors[120]
            c_nn.Conv2d(32, 64, kernel_size=3, padding=1, dtype=torch.complex64),
            c_nn.BatchNorm2d(64),
            act,
            c_nn.AvgPool2d(2),
        )

        self.classifier = nn.Sequential(
            c_nn.Flatten(),
            c_nn.Linear(64 * 7 * 7, 128, dtype=torch.complex64),
            act,
            c_nn.Linear(128, 10, dtype=torch.complex64),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # For classification, map complex logits to real via magnitude or real part.
        return x.abs()  # shape [B, 10], real-valued

# Example dataset stub: replace with the actual Fourier MNIST loader from torchcvnn examples.
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

def train_one_epoch(model, loader, optimizer, criterion, epoch, act_name):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
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

def evaluate(model, loader, criterion, act_name):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
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

def main():
    import torchvision
    import torchvision.transforms as T

    # 1) load standard MNIST
    transform = T.Compose([
        T.ToTensor(),  # [0,1] float, shape [1,28,28]
    ])
    train_real = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_real  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # 2) wrap with Fourier-complex dataset
    train_ds = ComplexFourierMNIST(train_real)
    test_ds  = ComplexFourierMNIST(test_real)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # 3) define the set of complex activations you want to test
    activations_to_test = ["modrelu", "zrelu", "cardioid", "c_relu"]

    for act_name in activations_to_test:
        print(f"\n=== Testing activation: {act_name} ===")
        model = ComplexMnistCNN(act_name=act_name).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, 6):
            train_one_epoch(model, train_loader, optimizer, criterion, epoch, act_name)
            evaluate(model, test_loader, criterion, act_name)

if __name__ == "__main__":
    main()
