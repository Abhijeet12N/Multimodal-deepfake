"""Quick 1-batch smoke test for multimodal training pipeline."""
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
for path in (HERE, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import torch
from train_multimodal import (
    DEVICE,
    EMBED_DIM,
    MultiTaskDetector,
    make_loader,
    multitask_loss,
)

WORK = os.environ.get("WORK_DIR", "/content/work")
meta_path = os.path.join(WORK, "meta_train.csv")

if not os.path.exists(meta_path):
    print(f"[smoke_test] {meta_path} not found. Run dataset scan first (python training/train_multimodal.py --stage scan).")
    sys.exit(0)

df = pd.read_csv(meta_path).head(48)
print("sample n=", len(df), "cats:", df.category.value_counts().to_dict())

loader = make_loader(df, augment=False, batch_size=8, num_workers=2, balanced=False)
model = MultiTaskDetector(EMBED_DIM).to(DEVICE)
print("trainable params:", round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2), "M")

t = time.time()
n = 0
for b in loader:
    for k in ["faces", "lips", "audio", "mel", "y_video", "y_audio", "y_any"]:
        b[k] = b[k].to(DEVICE)
    if DEVICE.type == "cuda":
        with torch.cuda.amp.autocast():
            preds = model(b["faces"], b["audio"], b["lips"], b["mel"])
            loss, parts = multitask_loss(preds, b)
    else:
        preds = model(b["faces"], b["audio"], b["lips"], b["mel"])
        loss, parts = multitask_loss(preds, b)
    print("batch ok | faces=", tuple(b["faces"].shape), "loss=", round(loss.item(), 4), "parts=", parts)
    n += 1
    if n >= 3:
        break
elapsed = time.time() - t
print("3 batches in", round(elapsed, 1), "s =>", round(elapsed / 3, 1), "s/batch (with first-batch warmup)")
