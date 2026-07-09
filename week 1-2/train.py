"""Train the toy ViT on CIFAR-10.

Usage:
    python train.py                 # sensible defaults (10 epochs)
    python train.py --epochs 30     # train longer for higher accuracy
    python train.py --smoke         # 1 epoch on a small subset (quick check)

Expects the CIFAR-10 dataset in ImageFolder layout under ./data/cifar10:
    data/cifar10/train/<class>/*.png
    data/cifar10/test/<class>/*.png
Run `python get_data.py` once to download and extract it.
Runs on Apple MPS / CUDA / CPU automatically.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from vit import ViT, ViTConfig

DATA_ROOT = "./data/cifar10"

# CIFAR-10 channel statistics (standard values).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(batch_size: int, smoke: bool):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_ROOT}. Run `python get_data.py` first."
        )

    train_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
    test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_tf)

    if smoke:
        # Shuffled indices so the subset spans multiple classes.
        g = torch.Generator().manual_seed(0)
        tr_idx = torch.randperm(len(train_set), generator=g)[:2000].tolist()
        te_idx = torch.randperm(len(test_set), generator=g)[:1000].tolist()
        train_set = Subset(train_set, tr_idx)
        test_set = Subset(test_set, te_idx)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2,
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def train(args):
    device = get_device()
    print(f"Device: {device}")

    train_loader, test_loader = build_loaders(args.batch_size, args.smoke)

    cfg = ViTConfig()
    model = ViT(cfg).to(device)
    print(f"Model: {model.num_params():,} params, {cfg.num_patches} patches/image")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = seen = 0

        for step, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss {running_loss / seen:.4f}")

        scheduler.step()
        acc = evaluate(model, test_loader, device)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:2d}/{args.epochs} | loss {running_loss / seen:.4f} "
              f"| test acc {acc:.2f}% | best {best_acc:.2f}% "
              f"| {time.time() - t0:.1f}s")

        if acc >= best_acc:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "acc": acc},
                "vit_cifar10.pt",
            )

    print(f"\nDone. Best test accuracy: {best_acc:.2f}%  (saved to vit_cifar10.pt)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--smoke", action="store_true", help="quick 1-epoch run on a subset")
    args = p.parse_args()
    if args.smoke:
        args.epochs = 1
        args.log_every = 10
    return args


if __name__ == "__main__":
    torch.manual_seed(0)
    train(parse_args())
