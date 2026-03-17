import os
import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from spine_dataset import build_datasets

torch.backends.cudnn.benchmark = True
#############################################
# CONFIG
#############################################

DATA_PATH = "/home/lhe09/projects/rsna_spine/outputs/roi_dataset_npy"

BATCH_SIZE = 32   # ConvNeXt-B는 메모리 많이 사용->ConvNeXt-Tiny로 교체
EPOCHS = 20
LR = 3e-4

DEVICE = "cuda"

SAVE_DIR = "/home/lhe09/projects/rsna_spine/outputs/results"
os.makedirs(SAVE_DIR, exist_ok=True)


#############################################
# CLASS WEIGHT (IMBALANCE 대응)
#############################################

CLASS_WEIGHTS = torch.tensor(
    [0.1, 1.0, 2.0],
    dtype=torch.float32
).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)


#############################################
# MODEL
#############################################

class SpineModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",   # convnext_base는 메모리 너무 많이 사용
            pretrained=True,
            num_classes=0
        )

        feat_dim = 768  # convnext_base는 1024

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.head = nn.Linear(1024, 15)

    def forward(self, x):

        B,S,C,H,W = x.shape

        x = x.view(B*S,C,H,W)

        feat = self.backbone(x)

        feat = feat.view(B,S,-1)

        lstm_out,_ = self.lstm(feat)

        feat = lstm_out[:,-1]

        out = self.head(feat)

        out = out.view(B,5,3)

        return out


#############################################
# LOSS
#############################################

def compute_loss(pred, label):

    loss = 0

    for i in range(5):
        loss += criterion(pred[:,i,:], label[:,i])

    return loss / 5


#############################################
# METRICS
#############################################

def compute_metrics(y_true, y_pred, y_prob):

    acc = accuracy_score(y_true, y_pred)

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr"
    )

    return acc, f1, auc


#############################################
# VALIDATION
#############################################

def validate(model, loader):

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for batch in tqdm(loader):

            img = batch["image"].to(DEVICE)
            label = batch["label"].to(DEVICE)

            pred = model(img)

            prob = torch.softmax(pred, dim=-1)

            pred_cls = torch.argmax(pred, dim=-1)

            all_preds.append(pred_cls.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(prob.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    acc_list = []
    f1_list = []
    auc_list = []

    for i in range(5):

        acc,f1,auc = compute_metrics(
            all_labels[:,i],
            all_preds[:,i],
            all_probs[:,i]
        )

        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)

    return np.mean(acc_list), np.mean(f1_list), np.mean(auc_list)


#############################################
# TRAIN
#############################################

def train():

    train_dataset, val_dataset = build_datasets(DATA_PATH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = SpineModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.5
    )

    best_auc = 0

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for batch in tqdm(train_loader):

            img = batch["image"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            pred = model(img)
            loss = compute_loss(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        print(f"\nEpoch {epoch}")
        print("Train Loss:", train_loss)

        acc,f1,auc = validate(model,val_loader)

        scheduler.step()

        torch.save({"epoch": epoch, 
                    "model": model.state_dict(), 
                    "optimizer": optimizer.state_dict()
                    }, f"checkpoint_{epoch}.pt") # checkpoint

        print("VAL ACC:",acc)
        print("VAL F1:",f1)
        print("VAL AUC:",auc)
        print("LR:", optimizer.param_groups[0]["lr"])

        if auc > best_auc:

            best_auc = auc

            torch.save(
                model.state_dict(),
                os.path.join(SAVE_DIR,"best_model_convnext_lstm.pth")
            )

            print("Best model saved")


#############################################
# MAIN
#############################################

if __name__ == "__main__":

    train()