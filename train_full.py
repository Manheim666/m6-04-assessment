from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
INITIAL_WEIGHTS = ROOT / "yolo26s.pt"
DATA = ROOT / "data.yaml"
PROJECT = ROOT / "runs" / "full"

EPOCHS = 150
BATCH = 16
IMGSZ = 960
WORKERS = 8

model = YOLO(str(INITIAL_WEIGHTS))
results = model.train(
    data=str(DATA),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    workers=WORKERS,
    project=str(PROJECT),
    name="run",
    exist_ok=True,
    cache="ram",
    device=0,
    patience=40,
    save=True,
    cos_lr=True,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    weight_decay=0.0005,
    mosaic=1.0,
    close_mosaic=15,
    mixup=0.15,
    copy_paste=0.3,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    amp=True,
)

best = PROJECT / "run" / "weights" / "best.pt"
print(f"\n=== Final validation ===")
model = YOLO(str(best))
metrics = model.val(data=str(DATA), imgsz=IMGSZ, batch=BATCH)
print(f"Final mAP50: {metrics.box.map50:.4f}")
print(f"Final mAP50-95: {metrics.box.map:.4f}")
print(f"Best weights: {best}")
