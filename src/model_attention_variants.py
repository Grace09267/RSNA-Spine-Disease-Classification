import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Attention Block ---
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn_map = self.attn(x)  # (B, 1, H, W)
        return x * attn_map

# --- Backbone ---
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

# --- Coord Regressor ---
class CoordRegressor(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 각 슬라이스에 대해 (x, y)
        )

    def forward(self, x):
        return self.fc(x)

# --- Classifier ---
class Classifier(nn.Module):
    def __init__(self, input_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 25 * 3)
        )

    def forward(self, x):
        return self.fc(x).view(-1, 25, 25, 3)

# --- Coord Projection ---
class CoordProjector(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.coord_proj = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 16)
        )

    def forward(self, coord):
        return self.coord_proj(coord.view(coord.size(0), -1))  # (B, 50) -> (B, 16)

# --- 모델 1: Segmentation 기반 Attention + Coord ---
class AttentionClassifier_Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.attn = AttentionBlock(1)
        self.coord_proj = CoordProjector()
        self.classifier = Classifier(input_dim=128 + 16)
        self.coord_regressor = CoordRegressor()

    def forward(self, x_img, x_heat, x_seg, x_coord):
        x_attn = self.attn(x_seg)
        coord_feat = self.coord_proj(x_coord)  # (B, 16)
        #print("[forward] coord_feat shape:", coord_feat.shape)
        coord_feat = coord_feat.unsqueeze(1).repeat(1, 25, 1).view(-1, 16)
        #print("[forward] coord_feat after repeat+view:", coord_feat.shape)    
        x_feat = self.backbone(x_img * x_attn)  # (B*25, 128)
       # print("[forward] x_feat shape:", x_feat.shape)
        fused = torch.cat([x_feat, coord_feat], dim=1)
        return self.classifier(fused), self.coord_regressor(x_feat)

# --- 모델 2: Heatmap 기반 Attention + Coord ---
class AttentionClassifier_Heat(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.attn = AttentionBlock(1)
        self.coord_proj = CoordProjector()
        self.classifier = Classifier(input_dim=128 + 16)
        self.coord_regressor = CoordRegressor()

    def forward(self, x_img, x_heat, x_seg, x_coord):
        x_attn = self.attn(x_heat)
        coord_feat = self.coord_proj(x_coord)
        coord_feat = coord_feat.unsqueeze(1).repeat(1, 25, 1).view(-1, 16)
        x_feat = self.backbone(x_img * x_attn)
        fused = torch.cat([x_feat, coord_feat], dim=1)
        return self.classifier(fused), self.coord_regressor(x_feat)

# --- 모델 3: Segmentation + Heatmap Fusion + Coord ---
class AttentionClassifier_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.attn = AttentionBlock(1)
        self.coord_proj = CoordProjector()
        self.classifier = Classifier(input_dim=128 + 16)
        self.coord_regressor = CoordRegressor()

    def forward(self, x_img, x_heat, x_seg, x_coord):
        x_fused = (x_heat + x_seg) / 2
        x_attn = self.attn(x_fused)
        coord_feat = self.coord_proj(x_coord)
        coord_feat = coord_feat.unsqueeze(1).repeat(1, 25, 1).view(-1, 16)
        x_feat = self.backbone(x_img * x_attn)
        fused = torch.cat([x_feat, coord_feat], dim=1)
        return self.classifier(fused), self.coord_regressor(x_feat)
