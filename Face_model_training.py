import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms, models
import random

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_DIR        = 'Dataset/Dataset'
ACTORS_FILE     = 'actors.txt'
BATCH_SIZE      = 32
EPOCHS          = 80          # more epochs; cosine schedule uses them all
EMBEDDING_SIZE  = 512
IMAGE_SIZE      = 112
WEIGHT_DECAY    = 0.01
DROPOUT_RATE    = 0.45
COSINE_SCALE    = 32          # raised from 20 → sharper angular decision boundary
VAL_SPLIT       = 0.15
SEED            = 42

# ── Learning rate schedule ────────────────────────────────────────────────
# Warmup for 5 epochs, then cosine anneal to LR_MIN.
# Prevents the aggressive early LR decay that killed learning after epoch 38.
LR_MAX          = 3e-4
LR_MIN          = 1e-6
WARMUP_EPOCHS   = 5

# ── Backbone freezing ─────────────────────────────────────────────────────
# Was 15 (only 4 blocks open). Now 10 → more high-level features fine-tuned.
# Still safe because strong augmentation prevents overfitting those layers.
UNFREEZE_FROM   = 10

# ── MixUp ────────────────────────────────────────────────────────────────
MIXUP_ALPHA     = 0.3         # Beta distribution concentration; 0 = off


# ─────────────────────────────────────────────
# Custom augmentation transforms
# ─────────────────────────────────────────────
class RandomGaussianBlur:
    def __init__(self, p=0.20, radius_range=(0.3, 1.2)):
        self.p = p
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(
                radius=random.uniform(*self.radius_range)))
        return img


class RandomOcclusion:
    """Black rectangle patch simulating glasses / hands / scarves."""
    def __init__(self, p=0.30, patch_scale=(0.05, 0.18)):
        self.p = p
        self.patch_scale = patch_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img
        from PIL import ImageDraw
        img  = img.copy()
        w, h = img.size
        pw = int(random.uniform(*self.patch_scale) * w)
        ph = int(random.uniform(*self.patch_scale) * h)
        x0 = random.randint(0, max(0, w - pw))
        y0 = random.randint(0, max(0, h - ph))
        ImageDraw.Draw(img).rectangle([x0, y0, x0+pw, y0+ph], fill=(0,0,0))
        return img


class RandomLighting:
    """Directional gradient overlay → lighting direction invariance."""
    def __init__(self, p=0.35, intensity=0.30):
        self.p = p
        self.intensity = intensity

    def __call__(self, img):
        if random.random() > self.p:
            return img
        from PIL import Image as PILImage
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        lo, hi = 1 - self.intensity, 1 + self.intensity
        direction = random.choice(['left', 'right', 'top', 'bottom'])
        if direction == 'left':
            mask = np.tile(np.linspace(lo, hi, w), (h, 1))
        elif direction == 'right':
            mask = np.tile(np.linspace(hi, lo, w), (h, 1))
        elif direction == 'top':
            mask = np.tile(np.linspace(lo, hi, h).reshape(-1,1), (1, w))
        else:
            mask = np.tile(np.linspace(hi, lo, h).reshape(-1,1), (1, w))
        arr = np.clip(arr * mask[:,:,np.newaxis], 0, 255).astype(np.uint8)
        return PILImage.fromarray(arr)


# ─────────────────────────────────────────────
# Transform pipelines
# ─────────────────────────────────────────────
def build_train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05,0.05),
                                scale=(0.88, 1.12), shear=6),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.3, hue=0.08),
        transforms.RandomGrayscale(p=0.10),
        RandomLighting(p=0.35, intensity=0.30),
        RandomGaussianBlur(p=0.20),
        RandomOcclusion(p=0.30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        transforms.RandomErasing(p=0.25, scale=(0.02,0.12),
                                 ratio=(0.3,3.0), value=0),
    ])


def build_val_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    ])


# ─────────────────────────────────────────────
# MixUp
# ─────────────────────────────────────────────
def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float):
    """
    Linearly interpolates pairs of images and their one-hot labels.
    Forces the model to learn smooth decision boundaries rather than
    memorising individual face crops.
    Returns mixed images + (label_a, label_b, lambda).
    """
    if alpha <= 0:
        return imgs, labels, labels, 1.0

    lam   = np.random.beta(alpha, alpha)
    lam   = max(lam, 1 - lam)          # keep dominant image > 50 %
    idx   = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


def mixup_loss(criterion, logits, label_a, label_b, lam):
    return lam * criterion(logits, label_a) + (1 - lam) * criterion(logits, label_b)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, root_dir: str, actors_file: str = None):
        self.samples = []
        self.classes = []

        allowed = None
        if actors_file and os.path.exists(actors_file):
            with open(actors_file) as f:
                allowed = {l.strip() for l in f if l.strip()}
            print(f"Whitelist loaded : {len(allowed)} identities")

        identity_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        label_idx = 0
        for identity in identity_dirs:
            if allowed and identity not in allowed:
                continue
            img_paths = (
                glob.glob(os.path.join(root_dir, identity, "*.jpg"))  +
                glob.glob(os.path.join(root_dir, identity, "*.jpeg")) +
                glob.glob(os.path.join(root_dir, identity, "*.png"))
            )
            if not img_paths:
                continue
            self.classes.append(identity)
            for p in img_paths:
                self.samples.append((p, label_idx))
            label_idx += 1

        print(f"Dataset ready    : {len(self.classes)} identities, "
              f"{len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return Image.open(path).convert("RGB"), label


class TransformSubset(Dataset):
    """Wraps a Subset with its own independent transform pipeline."""
    def __init__(self, subset: Subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────
class FaceEmbeddingModel(nn.Module):
    """
    MobileNetV2 → GlobalAvgPool → Projection Head → 512-D unit embedding

    UNFREEZE_FROM=10 opens 9 blocks (was 4).  Combined with strong
    augmentation this gives the model more capacity without overfitting.
    """
    def __init__(self, embedding_size=512, dropout=0.45, unfreeze_from=10):
        super().__init__()
        base          = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = base.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))

        for i, layer in enumerate(self.backbone):
            if i < unfreeze_from:
                for p in layer.parameters():
                    p.requires_grad = False

        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total  = sum(1 for p in self.backbone.parameters())
        print(f"Backbone params  : {frozen}/{total} frozen  "
              f"(blocks 0–{unfreeze_from-1})")

        self.projection = nn.Sequential(
            nn.Linear(1280, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.PReLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)


class CosineLinear(nn.Module):
    """
    Logit = scale * cos(θ).
    scale raised to 32 → tighter angular clusters → better separation.
    """
    def __init__(self, in_features, num_classes, scale=32.0):
        super().__init__()
        self.scale  = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        w = F.normalize(self.weight, p=2, dim=1)
        return self.scale * F.linear(x, w)


class FaceRecognitionTrainer(nn.Module):
    def __init__(self, num_classes, embedding_size=512,
                 dropout=0.45, scale=32.0, unfreeze_from=10):
        super().__init__()
        self.backbone = FaceEmbeddingModel(embedding_size, dropout, unfreeze_from)
        self.head     = CosineLinear(embedding_size, num_classes, scale)

    def forward(self, x):
        emb    = self.backbone(x)
        logits = self.head(emb)
        return logits, emb


# ─────────────────────────────────────────────
# LR Schedule: linear warmup + cosine anneal
# ─────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs, total_epochs, lr_min, lr_max):
    """
    Epochs 1..warmup  : LR ramps linearly from lr_min → lr_max
    Epochs warmup..end: LR follows cosine curve from lr_max → lr_min

    This replaces ReduceLROnPlateau which was decaying too aggressively
    (reached 3.7e-5 by epoch 38 / 50, effectively stopping learning).
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs          # linear ramp
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1 + np.cos(np.pi * progress))
        scaled   = lr_min + (lr_max - lr_min) * cosine
        return scaled / lr_max                           # scheduler multiplies by lr_max

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # MixUp
        imgs, label_a, label_b, lam = mixup_batch(imgs, labels, mixup_alpha)

        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss      = mixup_loss(criterion, logits, label_a, label_b, lam)
        loss.backward()

        # Gradient clipping — prevents exploding gradients with more layers unfrozen
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += (logits.argmax(1) == label_a).sum().item()  # dominant label
        total        += imgs.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, _    = model(imgs)
        loss         = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        correct      += (logits.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

    return running_loss / total, correct / total


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device     : {DEVICE}")
    print(f"CUDA available   : {torch.cuda.is_available()}")

    # ── Dataset ────────────────────────────────────────────────────────────
    base_dataset = FaceDataset(DATA_DIR, ACTORS_FILE)
    num_classes  = len(base_dataset.classes)
    print(f"Number of classes: {num_classes}")

    val_size   = int(len(base_dataset) * VAL_SPLIT)
    train_size = len(base_dataset) - val_size
    train_sub, val_sub = random_split(
        base_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_ds = TransformSubset(train_sub, build_train_transform(IMAGE_SIZE))
    val_ds   = TransformSubset(val_sub,   build_val_transform(IMAGE_SIZE))
    print(f"Train samples    : {len(train_ds)}")
    print(f"Val   samples    : {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = FaceRecognitionTrainer(
        num_classes    = num_classes,
        embedding_size = EMBEDDING_SIZE,
        dropout        = DROPOUT_RATE,
        scale          = COSINE_SCALE,
        unfreeze_from  = UNFREEZE_FROM,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total_p:,} "
          f"({100*trainable/total_p:.1f}%)")

    # ── Loss / Optimiser / Scheduler ───────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = LR_MAX,
        weight_decay = WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS, LR_MIN, LR_MAX)

    # ── Training loop ──────────────────────────────────────────────────────
    history       = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
    best_val_loss = float('inf')
    best_val_acc  = 0.0

    print(f"\n{'─'*65}")
    print(f"  Training  |  {EPOCHS} epochs  |  {num_classes} classes  |  {DEVICE}")
    print(f"  Warmup: {WARMUP_EPOCHS} epochs  |  MixUp α={MIXUP_ALPHA}  "
          f"|  Unfreeze from block {UNFREEZE_FROM}")
    print(f"{'─'*65}\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, MIXUP_ALPHA)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE)

        scheduler.step()   # cosine schedule steps every epoch (not on plateau)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]['lr']
        tag    = ""

        # Save on best val loss OR best val accuracy (whichever is new record)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss'            : val_loss,
                'val_acc'             : val_acc,
                'classes'             : base_dataset.classes,
                'embedding_size'      : EMBEDDING_SIZE,
            }, 'best_face_model.pth')
            tag = "  ✔ best"

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_acc_model.pth')
            tag += f"  (best acc {best_val_acc:.4f})"

        print(f"Epoch [{epoch:03d}/{EPOCHS}]  "
              f"Train {train_loss:.4f}/{train_acc:.4f}  "
              f"Val {val_loss:.4f}/{val_acc:.4f}  "
              f"LR {lr_now:.6f}{tag}")

    print(f"\n── Final Results ──────────────────────────────────────────")
    print(f"  Best val loss : {best_val_loss:.4f}")
    print(f"  Best val acc  : {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")

    torch.save(model.backbone.state_dict(), 'face_embedding_backbone.pth')
    print("\nBackbone    → face_embedding_backbone.pth")
    print("Best loss   → best_face_model.pth")
    print("Best acc    → best_acc_model.pth")

    plot_history(history, best_val_acc)
    print("Curves      → training_curves.png")


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_history(history: dict, best_acc: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   linewidth=2)
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   linewidth=2)
    axes[1].axhline(best_acc, color='red', linestyle='--',
                    linewidth=1, label=f"Best val {best_acc:.3f}")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.suptitle("Face Recognition Fine-tuning", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────
@torch.no_grad()
def get_embedding(backbone: FaceEmbeddingModel,
                  img_path: str,
                  device: torch.device) -> np.ndarray:
    """512-D unit-norm face embedding for a single crop."""
    img = Image.open(img_path).convert("RGB")
    img = build_val_transform(IMAGE_SIZE)(img).unsqueeze(0).to(device)
    backbone.eval()
    return backbone(img).cpu().numpy().squeeze()


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Dot product of unit-norm embeddings = cosine similarity.
    Recommended same-person threshold: > 0.40
    """
    return float(np.dot(emb1, emb2))


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()