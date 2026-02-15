# === attention_fusion_model.py ===
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from model import AttentionClassifier
from dataset import HeatmapCoordDataset

# 설정
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# 데이터 경로
full_df = pd.read_csv("rsna_2024_spine/slice_level_labeled.csv")
image_dir = "rsna_2024_spine/train_images"
heatmap_dir = "rsna_2024_spine/heatmaps"
coord_dir = "rsna_2024_spine/coords"

# 분할
train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
train_dataset = HeatmapCoordDataset(train_df, image_dir, heatmap_dir, coord_dir)
val_dataset = HeatmapCoordDataset(val_df, image_dir, heatmap_dir, coord_dir)

def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch])
    }

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# 모델
model = AttentionClassifier().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)
criterion_coord = torch.nn.MSELoss()

best_auc = 0
patience = 5
patience_counter = 0
num_diseases = 25
num_slices = 25

for epoch in range(50):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        x_img = batch["image"].cuda().view(-1, 1, 512, 512)
        x_heat = batch["heatmap"].cuda().view(-1, 1, 512, 512)
        x_coord = batch["coords"].cuda()
        y = batch["label"].cuda()  # (B, 25, 25)

        optimizer.zero_grad()
        logits, pred_coord = model(x_img, x_heat, x_coord)

        # 분류 손실
        loss_cls = 0.0
        valid_count = 0
        for i in range(num_slices):
            for d in range(num_diseases):
                valid_mask = (y[:, i, d] != -1)
                if valid_mask.any():
                    loss_cls += criterion_cls(logits[:, i, d, :][valid_mask], y[:, i, d][valid_mask])
                    valid_count += 1
        loss_cls = loss_cls / valid_count if valid_count > 0 else 0.0

        # 좌표 회귀 손실
        loss_coord = criterion_coord(pred_coord, x_coord.view(x_coord.size(0), -1, 2))
        loss = loss_cls + 0.2 * loss_coord
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss/len(train_loader):.4f}")

    # === 검증 ===
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            x_img = batch["image"].cuda().view(-1, 1, 512, 512)
            x_heat = batch["heatmap"].cuda().view(-1, 1, 512, 512)
            x_coord = batch["coords"].cuda()
            y = batch["label"].cuda()

            logits, _ = model(x_img, x_heat, x_coord)
            prob = F.softmax(logits, dim=-1)  # (B, 25, 25, 3)

            for i in range(num_slices):
                for d in range(num_diseases):
                    valid_mask = (y[:, i, d] != -1)
                    if valid_mask.any():
                        preds.extend(prob[:, i, d, :][valid_mask].cpu().numpy())
                        targets.extend(F.one_hot(y[:, i, d][valid_mask], num_classes=3).cpu().numpy())

    try:
        auc = roc_auc_score(targets, preds, average='macro')
    except:
        auc = 0.0
    print(f"Validation AUC: {auc:.4f}")

    # === 모델 저장 ===
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), os.path.join(save_dir, "best_attention_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping.")
            break
