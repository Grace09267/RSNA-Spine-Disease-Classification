import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from spine_dataset import build_datasets
from torch.amp import autocast, GradScaler

import random
import torch

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

torch.backends.cudnn.benchmark = True
#############################################
# CONFIG
#############################################

DATA_PATH = "/home/lhe09/projects/rsna_spine/outputs/roi_dataset_npy"

BATCH_SIZE = 16   # ConvNeXt-B는 메모리 많이 사용->ConvNeXt-Tiny로 교체
EPOCHS = 20
LR = 1e-5

DEVICE = "cuda"

SAVE_DIR = "/home/lhe09/projects/rsna_spine/outputs/results"
os.makedirs(SAVE_DIR, exist_ok=True)


#############################################
# CLASS WEIGHT (IMBALANCE 대응)
#############################################

CLASS_WEIGHTS = torch.tensor(
    [0.2, 1.0, 2.0],
    dtype=torch.float32
).to(DEVICE)

#criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
#criterion = nn.CrossEntropyLoss()

# ---------- focalloss로 ACU는 높은데 ACC가 너무 낮게 나옴으로 probility를 부드럽게 해줄, label smooth로 바꿈
#class FocalLoss(nn.Module):
#    def __init__(self, alpha=None, gamma=2):
#        super().__init__()
#        self.alpha = alpha
#        self.gamma = gamma
#        self.ce = nn.CrossEntropyLoss(weight=alpha)
#
#    def forward(self, logits, targets):
#        ce_loss = self.ce(logits, targets)
#        pt = torch.exp(-ce_loss)
#        focal_loss = (1 - pt) ** self.gamma * ce_loss
#        return focal_loss

#criterion = FocalLoss(alpha=CLASS_WEIGHTS, gamma=2)
#-------
criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=0.05)
#criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

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

        # 첫 conv layer 수정 (3채널 → 1채널)
        old_conv = self.backbone.stem[0]

        self.backbone.stem[0] = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        # pretratined weight 평균으로 초기화
        self.backbone.stem[0].weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        feat_dim = 768  # convnext_base는 1024

        # 🔥 positional embedding (learnable)
        #self.pos_embed = nn.Parameter(torch.randn(1, 30, feat_dim))  # S 최대 30 가정
        #self.pos_embed = nn.Parameter(torch.zeros(1, 101, feat_dim)) # randn이 너무 노이즈가 많은 것 같아 zero로
        # 🔥 Transformer Encoder : 디스크 간 관계 학습, 전체 구조 파악
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=4,
            dim_feedforward=512, # 내부연산 차원이라서 입력 dim이랑 다름
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1   # 👉 처음은 2로 시작 (안정적)
        )

        self.dropout = nn.Dropout(0.3)

        self.head = nn.Linear(feat_dim, 15)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.norm = nn.LayerNorm(feat_dim)
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, x):
        B,S,C,H,W = x.shape
        # CNN
        x = x.view(B*S,C,H,W)
        feat = self.backbone(x)
        feat = feat.view(B,S,-1)
        # layerNorm
        feat = self.norm(feat)

        # 🔥 positional encoding 추가
        #feat = feat + self.pos_embed[:, :S+1, :] # Transformer에서 "순서 정보" 알려줌

        # Transformer
        feat = self.transformer(feat)   # (B, S, 768)

        # attention pooing
        attn_weight = torch.softmax(self.attn(feat), dim=1)
        feat = (feat * attn_weight).sum(dim=1)
        
        # pooling (mean) CLS token 없이 간단하게 안정적이라서
        #feat = feat.mean(dim=1)

        # CLS 추가
        #cls_token = self.cls_token.expand(B, -1, -1)
        #feat = torch.cat([cls_token, feat], dim=1) # (B, S+1, C)
        ## CLS 만 사용
        #feat = feat[:, 0] # (B, C)

        feat = self.dropout(feat)
        # head
        out = self.head(feat)
        out = out.view(B,5,3)
        return out, attn_weight


#############################################
# LOSS
#############################################
DISC_WEIGHTS = torch.tensor([1.0, 1.0, 1.2, 1.5, 1.5]).to(DEVICE)

def compute_loss(pred, label):
    loss = 0
    for i in range(5):
        #loss += DISC_WEIGHTS[i] * criterion(pred[:,i,:], label[:,i])
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
            logits = pred / 0.7 # temperature(확률을 더 shape하게 만들는 방법)
            prob = torch.softmax(logits, dim=-1)
            #prob = torch.softmax(pred, dim=-1)
            pred_cls = torch.argmax(pred, dim=-1) # AUC값은 높은데, 정확도가 낮아 argmax보다 threshold를 쓰려고함
            #pred_cls = torch.zeros_like(label)
            # severe 먼저 잡기(class 2)
            #pred_cls[prob[:,:,2] > 0.45] = 2
            # moderate(class 1)
            #pred_cls[(prob[:,:,1] > 0.55) & (pred_cls !=2)] = 1

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

    scaler = GradScaler("cuda")

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for batch in tqdm(train_loader):

            img = batch["image"].to(DEVICE, non_blocking=True)
            label = batch["label"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with autocast("cuda"):
                pred = model(img)
                loss = compute_loss(pred, label)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 학습중 갑자기 어느 에포크에서 train loss가 nan이나옴. lstm에서 gradient 폭발 안하기 위해 적어야함.
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach().item()
                                         

            #
            #print("pred unique:", torch.unique(pred_cls))
            #print("label unique:", torch.unique(label))
            #print("pred shape:", pred.shape)
            #print("pred[0]:", pred[0])
            #print(loss.item())
            #prob = torch.softmax(pred, dim=-1)               
            #print(prob[0])
            #print(label[0])
            #break
            #

        train_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch}")
        print("Train Loss:", train_loss)
        #
        #print("last batch loss:", loss.item())
        #print("running_loss:", running_loss)
        pred_cls = torch.argmax(pred, dim=-1)
        print(torch.unique(pred_cls))
        prob = torch.softmax(pred, dim=-1)
        print(prob.mean(dim=(0,1)))
        #

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
                os.path.join(SAVE_DIR,"best_model_convnext_transformer.pth")
            )

            print("Best model saved")


#############################################
# MAIN
#############################################

if __name__ == "__main__":

    train()