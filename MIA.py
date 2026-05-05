import os
import sys
import torch
import pandas as pd
import requests
import random
import argparse

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.models import resnet18
import torchvision.transforms as transforms


# config
BASE = Path(__file__).parent
PUB_PATH = BASE / "pub.pt"
PRIV_PATH = BASE / "priv.pt"
MODEL_PATH = BASE / "model.pt"
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "568be0178160b1148f06f177d7d56b9a"
TASK_ID = "01-mia"  #DONOT CHANGE



# dataset classes
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]


# load datasets
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)


# normalization (same as training)
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1491, 0.1864, 0.1301]

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])

pub_ds.transform = transform
priv_ds.transform = transform


# load model
print("Loading model...")
model = resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
model.maxpool = torch.nn.Identity()
model.fc = torch.nn.Linear(512, 9)

# select device: prefer MPS on Apple Silicon, then CUDA, else CPU
device = "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# load model onto device
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


# create random submission (remove this later or it will rewrite your actual submission)
print("Running membership attack (confidence-based, calibrated with public data if available)...")

def compute_confidences(dataset, model, device="cpu", batch_size=64):
    # Some saved Dataset objects may contain None entries that break default_collate.
    # Iterate samples manually and build batches to avoid collate errors.
    confidences = []
    ids = []
    imgs_batch = []
    ids_batch = []

    for idx in range(len(dataset)):
        try:
            sample = dataset[idx]
        except Exception:
            continue

        # unpack sample whether MembershipDataset or TaskDataset
        if sample is None:
            continue
        if len(sample) == 4:
            sid, img, label, _ = sample
        else:
            sid, img, label = sample

        if img is None:
            continue

        # If image is not a tensor, try to convert using dataset.transform if possible,
        # otherwise fall back to ToTensor.
        if not isinstance(img, torch.Tensor):
            if hasattr(dataset, "transform") and dataset.transform is not None:
                try:
                    img = dataset.transform(img)
                except Exception:
                    try:
                        img = transforms.ToTensor()(img)
                    except Exception:
                        continue
            else:
                try:
                    img = transforms.ToTensor()(img)
                except Exception:
                    continue

        imgs_batch.append(img.unsqueeze(0).to(device))
        ids_batch.append(str(sid))

        if len(imgs_batch) >= batch_size:
            imgs_t = torch.cat(imgs_batch, dim=0)
            with torch.no_grad():
                logits = model(imgs_t)
                probs = torch.softmax(logits, dim=1)
                conf, _ = probs.max(dim=1)
            confidences.extend(conf.cpu().numpy().tolist())
            ids.extend(ids_batch)
            imgs_batch = []
            ids_batch = []

    # flush remainder
    if len(imgs_batch) > 0:
        imgs_t = torch.cat(imgs_batch, dim=0)
        with torch.no_grad():
            logits = model(imgs_t)
            probs = torch.softmax(logits, dim=1)
            conf, _ = probs.max(dim=1)
        confidences.extend(conf.cpu().numpy().tolist())
        ids.extend(ids_batch)

    return ids, confidences


# compute confidences for public set (for calibration) and private set (to attack)
pub_ids, pub_conf = compute_confidences(pub_ds, model, device=device)
priv_ids, priv_conf = compute_confidences(priv_ds, model, device=device)

# try to get membership labels from public dataset for calibration
calibrated = False
if hasattr(pub_ds, "membership") and len(pub_ds.membership) == len(pub_conf):
    print("Calibrating scores using public dataset membership labels...")
    y = torch.tensor(pub_ds.membership, dtype=torch.float32).view(-1, 1)
    x = torch.tensor(pub_conf, dtype=torch.float32).view(-1, 1)

    # simple Platt scaling: learn a*x + b, optimized with BCE
    device = "cpu"
    model_cal = torch.nn.Linear(1, 1).to(device)
    optimizer = torch.optim.Adam(model_cal.parameters(), lr=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x_t = x.to(device)
    y_t = y.to(device)
    for epoch in range(200):
        optimizer.zero_grad()
        logits = model_cal(x_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()

    def calibrate(conf_list):
        with torch.no_grad():
            inp = torch.tensor(conf_list, dtype=torch.float32).view(-1, 1)
            out = torch.sigmoid(model_cal(inp)).view(-1).cpu().numpy().tolist()
        return out

    calibrated = True
else:
    print("Public dataset has no membership labels; using raw confidence as score.")
    def calibrate(conf_list):
        # Identity mapping
        return conf_list


# produce scores for private dataset
scores = calibrate(priv_conf)

df = pd.DataFrame({
    "id": priv_ids,
    "score": scores,
})

df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)


# submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)