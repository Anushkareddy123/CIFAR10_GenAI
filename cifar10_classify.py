#!/usr/bin/env python3
import os
import io
import base64
import random
import json
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image
import torch
import torchvision
from torchvision.datasets import CIFAR10
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ---------- OUTPUT FOLDER ----------
output_folder = os.path.abspath(".")
os.makedirs(output_folder, exist_ok=True)  # ensure folder exists

# ---------- LOAD API SECRETS ----------
load_dotenv(os.path.join(os.path.expanduser("~"), ".soonerai.env"))
API_KEY = os.getenv("SOONERAI_API_KEY")
BASE_URL = os.getenv("SOONERAI_BASE_URL", "https://ai.sooners.us").rstrip("/")
MODEL = os.getenv("SOONERAI_MODEL", "gemma3:4b")
if not API_KEY:
    raise RuntimeError("Missing SOONERAI_API_KEY in ~/.soonerai.env")

# ---------- CONFIG ----------
SEED = 1337
SAMPLES_PER_CLASS = 10
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

SYSTEM_PROMPT = """
You are a computer vision expert trained on the CIFAR-10 dataset.
Each image is a 32x32 color photo of one object. 
Look carefully and classify it as exactly one of these:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck.
Respond with only the label name.


"""

USER_INSTRUCTION = f"""
Classify this CIFAR-10 image. Respond with exactly one label from this list:
{', '.join(CLASSES)}
Your reply must be just the label, nothing else.
""".strip()

# ---------- HELPERS ----------
def pil_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def post_chat_completion_image(image_data_url: str) -> str:
    url = f"{BASE_URL}/api/chat/completions"
    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_INSTRUCTION},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]}
        ]
    }
    resp = requests.post(url, headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"API error {resp.status_code}: {resp.text}")
    return resp.json()["choices"][0]["message"]["content"].strip()

def normalize_label(text: str) -> str:
    t = text.lower().strip()
    if t in CLASSES: return t
    for c in CLASSES:
        if c in t: return c
    return "__unknown__"

def stratified_sample_cifar10(root: str = "./data") -> List[Tuple[Image.Image,int]]:
    ds = CIFAR10(root=root, train=True, download=True)
    per_class: Dict[int, List[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):
        per_class[label].append(idx)
    random.seed(SEED)
    selected = []
    for label in range(10):
        chosen = random.sample(per_class[label], SAMPLES_PER_CLASS)
        for idx in chosen:
            img, tgt = ds[idx]
            selected.append((img, tgt))
    return selected

# ---------- MAIN ----------
def main():
    print("Preparing CIFAR-10 sample (100 images)...")
    samples = stratified_sample_cifar10()
    y_true, y_pred, bad = [], [], []

    print("Classifying images...")
    for i, (img, tgt) in enumerate(samples, start=1):
        data_url = pil_to_base64_jpeg(img)
        try:
            reply = post_chat_completion_image(data_url)
        except Exception as e:
            reply, pred_idx, pred_label = "__error__", -1, "__error__"
            print(f"[{i}/100] API error: {e}")
        else:
            pred_label = normalize_label(reply)
            pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1

        y_true.append(tgt)
        y_pred.append(pred_idx)
        print(f"[{i:03d}/100] true={CLASSES[tgt]:>10s} | pred={pred_label:>10s} | raw='{reply}'")

        if pred_idx == -1:
            bad.append({"i": i, "true": CLASSES[tgt], "raw_reply": reply})

    # ---------- Accuracy ----------
    y_pred_fixed = [p if p in range(10) else 9 for p in y_pred]
    acc = accuracy_score(y_true, y_pred_fixed)
    print(f"\nAccuracy: {acc*100:.2f}%")

    # ---------- Save confusion matrix ----------
    cm_path = os.path.join(output_folder, "confusion_matrix.png")
    cm = confusion_matrix(y_true, y_pred_fixed, labels=list(range(10)))
    plt.figure(figsize=(7.5,7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("CIFAR-10 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(10), CLASSES, rotation=45, ha="right")
    plt.yticks(range(10), CLASSES)
    for r in range(10):
        for c in range(10):
            plt.text(c,r,str(cm[r,c]),ha="center",va="center")
    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"Saved {cm_path}")

    # ---------- Save misclassifications ----------
    mis_path = os.path.join(output_folder, "misclassifications.jsonl")
    with open(mis_path,"w") as f:
        for row in bad:
            f.write(json.dumps(row)+"\n")
    print(f"Saved {len(bad)} misclassifications to {mis_path}")

if __name__=="__main__":
    main()
