# Pseudo Segmentation Mask 생성(히트맵 기반반)
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import os
from tqdm import tqdm

def generate_masks_from_heatmaps(heatmap_dir="heatmaps", mask_dir="masks", threshold=0.05):
    os.makedirs(mask_dir, exist_ok=True)
    for file in tqdm(os.listdir(heatmap_dir)):
        if file.endswith(".npy"):
            heatmap = np.load(os.path.join(heatmap_dir, file))
            norm_heatmap = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
            mask = (norm_heatmap > threshold).astype(np.uint8)
            # 윤곽선 다듬기 (선택)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            np.save(os.path.join(mask_dir, file.replace(".npy", ".npy")), mask)


# Segmentation Dataset 클래스 정의(이미지 + mask 쌍쌍)
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx].replace(".png", ".npy"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        mask = np.load(mask_path)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask
# Step 2: UNet 모델 정의
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(1, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 32))

        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.up2(bottleneck)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))

# Step 3: 학습 루프 정의
def train_unet(model, dataloader, epochs=10, lr=1e-3, patience=3, save_path="segmentation_model.pth"):
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, masks in progress_bar:
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

        # 성능 평가
        all_preds = []
        all_masks = []
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.cuda()
                preds = model(images)
                preds = (preds > 0.5).float().cpu()
                all_preds.append(preds)
                all_masks.append(masks)

        all_preds = torch.cat(all_preds, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        dice = dice_score(all_preds, all_masks)
        iou = iou_score(all_preds, all_masks)
        print(f"🧪 Evaluation - Dice: {dice:.4f}, IoU: {iou:.4f}")

        
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved (loss={best_loss:.6f}) to {save_path}")
        else:
            patience_counter += 1
            print(f"⚠️ No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("⛔ Early stopping triggered.")
                break
        

def dice_score(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()

def iou_score(pred, target, eps=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target - pred * target).sum(dim=(1,2,3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

    
# Step 4: Heatmap과 Segmentation 비교 시각화
import matplotlib.pyplot as plt

def compare_heatmap_and_mask(image_path, heatmap_path, pred_mask):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    heatmap = np.load(heatmap_path)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original Image")
    axs[1].imshow(heatmap, cmap='hot')
    axs[1].set_title("Heatmap")
    axs[2].imshow(pred_mask.squeeze(), cmap='gray')
    axs[2].set_title("UNet Predicted Mask")
    plt.show()

# -------------------------------
# Step 5. 실행
# -------------------------------
if __name__ == "__main__":
    generate_masks_from_heatmaps("heatmaps", "masks", threshold=0.05)
    dataset = SegmentationDataset("heatmaps", "masks")
    print(f"총 학습 샘플 수: {len(dataset)}")

    if len(dataset) == 0:
        print("❗ 데이터셋이 비어있습니다. 경로와 파일명을 확인하세요.")
    else:
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        model = UNet()
        train_unet(model, dataloader, epochs=10, lr=1e-3)
