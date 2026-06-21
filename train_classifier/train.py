# Scott Nguyen — MIT License
#
# Train the armor number classifier and export to ONNX.
#
# Expected dataset layout:
#   data/
#     1/        ← raw armor images for number 1
#     2/
#     3/
#     4/
#     5/
#     outpost/
#     guard/
#     base/
#     negative/ ← false positives / background patches
#
# Each image will be preprocessed the same way the C++ pipeline does:
#   1. Crop / warp to 20×28 grayscale
#   2. Otsu binary threshold
#   3. Normalize to [0, 1]
#
# Output:
#   ../armor_detector/model/mlp.onnx
#   ../armor_detector/model/label.txt

import os
import pathlib
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ── constants matching the C++ extractor ──────────────────────────────────────
IMG_W, IMG_H = 20, 28
LABELS = ["1", "2", "3", "4", "5", "outpost", "guard", "base", "negative"]


# ── dataset ───────────────────────────────────────────────────────────────────
def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_W, IMG_H))
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img.astype(np.float32) / 255.0


class ArmorDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []
        for label_idx, label in enumerate(LABELS):
            folder = pathlib.Path(root) / label
            if not folder.exists():
                print(f"[warn] missing class folder: {folder}")
                continue
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
                for p in folder.glob(ext):
                    self.samples.append((str(p), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = preprocess(path)
        x = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        return x, label


# ── model ─────────────────────────────────────────────────────────────────────
class ArmorMLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_W * IMG_H, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ── training loop ─────────────────────────────────────────────────────────────
def train(args):
    dataset = ArmorDataset(args.data)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {args.data}")

    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch)

    model = ArmorMLP(num_classes=len(LABELS))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)
        acc = correct / total if total else 0.0

        print(f"epoch {epoch:3d}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nBest val accuracy: {best_acc:.3f}")
    model.load_state_dict(torch.load("best_model.pt"))
    return model


# ── ONNX export ───────────────────────────────────────────────────────────────
def export_onnx(model: nn.Module, out_path: str):
    model.eval()
    dummy = torch.zeros(1, 1, IMG_H, IMG_W)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"Saved ONNX model → {out_path}")


def write_labels(out_path: str):
    with open(out_path, "w") as f:
        f.write("\n".join(LABELS))
    print(f"Saved labels → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train armor number classifier")
    parser.add_argument("--data", default="data", help="path to dataset root")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--out-dir",
        default=str(pathlib.Path(__file__).parent.parent / "armor_detector" / "model"),
        help="directory to write mlp.onnx and label.txt",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = train(args)
    export_onnx(model, os.path.join(args.out_dir, "mlp.onnx"))
    write_labels(os.path.join(args.out_dir, "label.txt"))
