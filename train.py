"""
Fine-tune VideoMAE's classification head (+ optionally unfreeze last blocks)
on UCF-Crime style data.

Usage (recommended -- uses the manifest built by prepare_data.py, which
matches the real UCF-Crime download layout):
    python prepare_data.py --root data
    python train.py --manifest data/manifest.csv --epochs 10 --batch_size 4 --freeze_backbone

Usage (legacy -- if you already reorganized videos into data/<Class>/*.mp4):
    python train.py --data_root data/UCF_Crimes/videos --epochs 10 --batch_size 4
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

import config
from dataset import UCFCrimeClipDataset, build_samples_from_folder, build_samples_from_manifest

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"


def collate(batch):
    clips, labels = zip(*batch)   # each clip: [T, H, W, C] uint8 RGB tensor
    return list(clips), torch.tensor(labels, dtype=torch.long)


def main(args):
    device = config.DEVICE
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(config.CRIME_CLASSES),
        ignore_mismatched_sizes=True,
    ).to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("[train] Backbone frozen -- training classifier head only.")

    if args.manifest:
        samples = build_samples_from_manifest(args.manifest)
        source_desc = args.manifest
    else:
        samples = build_samples_from_folder(args.data_root)
        source_desc = args.data_root

    if not samples:
        raise RuntimeError(
            f"No samples found from {source_desc}. If using --data_root, check the folder "
            "layout matches dataset.py. If using --manifest, re-run prepare_data.py and check "
            "its 'unmatched' warnings."
        )
    print(f"[train] Found {len(samples)} videos across {len(set(s[1] for s in samples))} classes.")

    dataset = UCFCrimeClipDataset(samples, clips_per_video=args.clips_per_video)
    val_len = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [train]"):
            inputs = processor([c.numpy() for c in clips], return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc = evaluate(model, processor, val_loader, device)
        print(f"[train] Epoch {epoch + 1}: loss={running_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "videomae_crime.pt")
            torch.save({"model_state_dict": model.state_dict(), "val_acc": val_acc}, ckpt_path)
            print(f"[train] Saved new best checkpoint -> {ckpt_path}")


@torch.no_grad()
def evaluate(model, processor, loader, device):
    model.eval()
    correct, total = 0, 0
    for clips, labels in tqdm(loader, desc="Validating", leave=False):
        inputs = processor([c.numpy() for c in clips], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        labels = labels.to(device)
        logits = model(pixel_values=pixel_values).logits
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=None,
                         help="path to manifest.csv produced by prepare_data.py (recommended)")
    parser.add_argument("--data_root", default=None,
                         help="legacy: path to a pre-organized data/<Class>/*.mp4 folder tree")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--clips_per_video", type=int, default=3,
                         help="random clips sampled per video, per epoch")
    parser.add_argument("--freeze_backbone", action="store_true",
                         help="train only the classification head (faster, less data-hungry)")
    args = parser.parse_args()
    main(args)
