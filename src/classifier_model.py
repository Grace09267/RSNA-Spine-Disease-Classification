#--------------------------------------------
# Backbone CNN: 예 ResNet18(이미지 + 히트맵 2채널 입력)
# 좌표 입력 : (x, y)를 flatten 후 FC layer를 통해 feature로 압축
# 두 feature concat 후 최종 분류기(Binary or multi-label)
#----------------------------------------------
# EfficientNet18-> EfficientNet-B0으로 수정
# === model.py ===
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch
import yaml

with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)

class AttentionClassifier(nn.Module):
    def __init__(self, num_slices=25, num_classes=3, num_diseases=25, dropout=0.3):
        super().__init__()
        self.num_slices = num_slices
        self.num_diseases = num_diseases

        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        self.backbone._conv_stem = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1, bias=False)

        self.coord_proj = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(), nn.Dropout(dropout)
        )

        self.shared_fc = nn.Sequential(
            nn.Linear(1280 + 16, 128), nn.ReLU(), nn.Dropout(dropout)
        )

        self.slice_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
            ) for _ in range(num_diseases)
        ])

    def forward(self, x_img, x_heat, x_coord):
        B = x_coord.shape[0]
        x_coord = x_coord.view(B, self.num_slices, 2)
        x = torch.cat([x_img, x_heat], dim=1)

        feat = self.backbone.extract_features(x)
        feat = nn.AdaptiveAvgPool2d(1)(feat).squeeze(-1).squeeze(-1)
        feat = feat.view(B, self.num_slices, -1)

        logits = torch.zeros((B, self.num_slices, self.num_diseases, 3), device=x.device)

        for i in range(self.num_slices):
            coord_feat = self.coord_proj(x_coord[:, i, :])
            fused = torch.cat([feat[:, i, :], coord_feat], dim=1)
            fused_fc = self.shared_fc(fused)
            for d in range(self.num_diseases):
                logits[:, i, d, :] = self.slice_classifiers[d](fused_fc)

        return logits, x_coord
