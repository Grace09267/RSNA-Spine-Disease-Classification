# attention_fusion_compare.py (DEBUG 버전)

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model_attention_variants import (
    AttentionClassifier_Seg,
    AttentionClassifier_Heat,
    AttentionClassifier_Fusion
)
from dataset import HeatmapCoordDataset

save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

image_dir = "rsna_2024_spine/train_images"
heatmap_dir = "heatmaps"
coord_dir = "coords"
segmask_dir = "masks"

full_df = pd.read_csv("slice_level_labeled_balanced_1000.csv")

# 필터링
full_df = full_df[full_df.apply(lambda row: 
    os.path.exists(os.path.join(coord_dir, f"{int(row['study_id'])}_{int(row['series_id'])}_{int(row['instance_number'])}.npy"))
    and any(row[col] != -1 for col in full_df.columns[3:]), axis=1)]

print(f"✅ 학습 가능한 유형 슬라이스 수: {len(full_df)}")

train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)
train_dataset = HeatmapCoordDataset(train_df, image_dir, heatmap_dir, coord_dir, segmask_dir)
val_dataset = HeatmapCoordDataset(val_df, image_dir, heatmap_dir, coord_dir, segmask_dir)

def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "heatmap": torch.stack([b["heatmap"] for b in batch]),
        "segmask": torch.stack([b["segmask"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch])
    }

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

print(f"[DEBUG] val_loader 길이: {len(val_loader)}")

model_dict = {
    "seg": AttentionClassifier_Seg,
    "heat": AttentionClassifier_Heat,
    "fusion": AttentionClassifier_Fusion
}

num_diseases = 25
num_slices = 25
patience = 5

for model_type, model_cls in model_dict.items():
    print(f"\n🧠 Training {model_type.upper()} Attention Model")
    model = model_cls().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_coord = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_auc = 0
    patience_counter = 0
    train_losses = []
    val_aucs = []

    for epoch in range(100):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"[{model_type.upper()}] Train Epoch {epoch+1}")):
            x_img = batch["image"].cuda().view(-1, 1, 512, 512)
            x_heat = batch["heatmap"].cuda().view(-1, 1, 512, 512)
            x_seg = batch["segmask"].cuda().view(-1, 1, 512, 512)
            x_coord = batch["coords"].cuda()
            y = batch["label"].cuda()

            if batch_idx == 0:
                print(f"[DEBUG] x_img.shape: {x_img.shape}")
                print(f"[DEBUG] x_coord.shape: {x_coord.shape}")
                print(f"[DEBUG] y.shape: {y.shape}")
                print(f"[DEBUG] unique labels in y: {torch.unique(y)}")
                print(f"[DEBUG] x_heat.shape: {x_heat.shape}")

            optimizer.zero_grad()
            logits, pred_coord = model(x_img, x_heat, x_seg, x_coord)

            loss_coord = criterion_coord(pred_coord.view(-1, 2), x_coord.view(-1, 2))

            logits_flat = logits.view(-1, 3)
            y_flat = y.view(-1)
            valid_mask = (y_flat != -1)

            loss_cls = criterion_cls(logits_flat[valid_mask], y_flat[valid_mask]) if valid_mask.sum() > 0 else torch.tensor(0.0, requires_grad=True).cuda()

            loss = loss_cls + 0.2 * loss_coord
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[{model_type.upper()}] Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        train_losses.append(avg_loss)

        # === 검증 ===
        model.eval()
        batch_aucs = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[{model_type.upper()}] Validating"):
                x_img = batch["image"].cuda().view(-1, 1, 512, 512)
                x_heat = batch["heatmap"].cuda().view(-1, 1, 512, 512)
                x_seg = batch["segmask"].cuda().view(-1, 1, 512, 512)
                x_coord = batch["coords"].cuda()
                y = batch["label"].cuda()

                #print(f"[DEBUG] y 전체 shape: {y.shape}, unique: {torch.unique(y)}, !=-1 count: {(y != -1).sum().item()}")

                logits, _ = model(x_img, x_heat, x_seg, x_coord)
                prob = F.softmax(logits, dim=-1)
                
                preds, targets = [], []

                for i in range(num_slices):
                    for d in range(num_diseases):
                        valid_mask = (y[:, i, d] != -1)
                        if valid_mask.any():
                            #preds.extend(prob[:, i, d, :][valid_mask].cpu().numpy())
                            #targets.extend(F.one_hot(y[:, i, d][valid_mask], num_classes=3).cpu().numpy())
                            pred_slice = prob[:, i, d, :][valid_mask].detach().cpu().numpy()
                            target_slice = F.one_hot(y[:, i, d][valid_mask], num_classes=3).cpu().numpy()
                            preds.append(pred_slice)
                            targets.append(target_slice)
                '''           
                if preds and targets:
                    try:
                        auc = roc_auc_score(
                            np.concatenate(targets, axis=0),
                            np.concatenate(preds, axis=0),
                            average='macro'
                        )
                        batch_aucs.append(auc)
                    except Exception as e:
                        print(f"⚠️ AUC 계산 실패 (batch): {e}")
                        
                '''
                # AUC 계산
                try:
                    y_true = np.concatenate(targets, axis=0)
                    y_pred = np.concatenate(preds, axis=0)
                    y_true_classes = np.argmax(y_true, axis=1)

                    #print(f"[DEBUG] y_true_classes 분포: {np.unique(y_true_classes, return_counts=True)}")
                    
                    if len(np.unique(y_true_classes)) < 2:
                        print("⚠️ Skip AUC calculation: only one class in y_true.")
                        batch_aucs.append(np.nan)
                        continue

                    auc = roc_auc_score(y_true, y_pred, average='macro')
                    if not np.isnan(auc):
                        batch_aucs.append(auc)
                        
                except Exception as e:
                    print(f"⚠️ AUC 계산 실패 (batch): {e}")
                
                            
                # 메모리 정리
                del x_img, x_heat, x_seg, x_coord, y, logits, prob
                torch.cuda.empty_cache()
                gc.collect()
        ''''
        if len(preds) == 0 or len(targets) == 0:
            print("❌ 유효한 검증 예측값이 없습니다. 건너뜁니다.")
            val_aucs.append(0.0)
            continue

        preds_np = np.array(preds)
        targets_np = np.array(targets)

        print(f"[DEBUG] preds_np.shape: {preds_np.shape}, targets_np.shape: {targets_np.shape}")

        if preds_np.ndim != 2 or targets_np.ndim != 2:
            print("❌ 예측값 차원이 올바르지 않습니다. 건너뜁니다.")
            val_aucs.append(0.0)
            continue

        pred_labels = np.argmax(preds_np, axis=1)
        true_labels = np.argmax(targets_np, axis=1)
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({model_type} attention)")
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_type}.png"))
        plt.close()
        '''
        valid_aucs = [x for x in batch_aucs if not np.isnan(x)]
        if len(valid_aucs) == 0:
            val_auc = 0.0
            print("❌ 유효한 validation batch AUC 없음 (모두 NaN)")
        else:
            val_auc = np.mean(valid_aucs)
            print(f"🧪 [{model_type.upper()}] Validation AUC (mean): {val_auc:.4f}")
            
        scheduler.step(val_auc)    
        val_aucs.append(val_auc)

        '''
        try:
            auc = roc_auc_score(targets_np, preds_np, average='macro')
        except Exception as e:
            print(f"❌ AUC 계산 실패: {e}")
            auc = 0.0

        print(f"🧪 [{model_type.upper()}] Validation AUC: {auc:.4f}")
        val_aucs.append(auc)
        '''
        if auc > best_auc:
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_{model_type}_model.pth"))
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔ Early stopping {model_type.upper()} model.")
                break

    np.save(os.path.join(save_dir, f"{model_type}_train_loss.npy"), train_losses)
    np.save(os.path.join(save_dir, f"{model_type}_val_auc.npy"), val_aucs)
