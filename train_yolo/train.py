# Scott Nguyen — MIT License
#
# Train a YOLOv8n armor plate detector and export to ONNX.
#
# Requirements:
#   pip install ultralytics
#
# Dataset layout (see dataset.yaml):
#   data/images/train/  ← images
#   data/labels/train/  ← YOLO format .txt annotations
#   data/images/val/
#   data/labels/val/
#
# After training, armor.onnx is written to:
#   ../armor_detector/model/armor.onnx
#
# The C++ detector automatically loads it at startup if present.
# Drop it in and rebuild — no other code changes needed.

import argparse
import pathlib
import shutil

from ultralytics import YOLO

REPO_ROOT = pathlib.Path(__file__).parent.parent
MODEL_OUT = REPO_ROOT / "armor_detector" / "model" / "armor.onnx"


def train(args):
    model = YOLO(args.weights)

    results = model.train(
        data=str(pathlib.Path(__file__).parent / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        lr0=args.lr,
        device=args.device,
        project="runs",
        name="armor_detector",
        exist_ok=True,
        # augmentation — helps generalize across lighting conditions
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.0,   # armor plates are not horizontally symmetric
        flipud=0.0,
        mosaic=1.0,
        mixup=0.1,
    )

    return model, results


def export(model, out_path: pathlib.Path):
    onnx_path = model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, out_path)
    print(f"Exported ONNX model → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 armor plate detector")
    parser.add_argument("--weights", default="yolov8n.pt", help="starting weights (pretrained)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="0", help="cuda device index or 'cpu'")
    parser.add_argument(
        "--out", default=str(MODEL_OUT), help="output path for armor.onnx"
    )
    args = parser.parse_args()

    model, _ = train(args)
    export(model, pathlib.Path(args.out))
