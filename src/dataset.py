# === dataset.py ===
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
disease_cols = [
    "spinal_canal_stenosis_l1_l2", "spinal_canal_stenosis_l2_l3", "spinal_canal_stenosis_l3_l4", "spinal_canal_stenosis_l4_l5", "spinal_canal_stenosis_l5_s1",
    "left_neural_foraminal_narrowing_l1_l2", "left_neural_foraminal_narrowing_l2_l3", "left_neural_foraminal_narrowing_l3_l4", "left_neural_foraminal_narrowing_l4_l5", "left_neural_foraminal_narrowing_l5_s1",
    "right_neural_foraminal_narrowing_l1_l2", "right_neural_foraminal_narrowing_l2_l3", "right_neural_foraminal_narrowing_l3_l4", "right_neural_foraminal_narrowing_l4_l5", "right_neural_foraminal_narrowing_l5_s1",
    "left_subarticular_stenosis_l1_l2", "left_subarticular_stenosis_l2_l3", "left_subarticular_stenosis_l3_l4", "left_subarticular_stenosis_l4_l5", "left_subarticular_stenosis_l5_s1",
    "right_subarticular_stenosis_l1_l2", "right_subarticular_stenosis_l2_l3", "right_subarticular_stenosis_l3_l4", "right_subarticular_stenosis_l4_l5", "right_subarticular_stenosis_l5_s1"
]

class HeatmapCoordDataset(Dataset):
    def __init__(self, df, image_dir, heatmap_dir, coord_dir, segmask_dir, num_slices=25):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.heatmap_dir = heatmap_dir
        self.coord_dir = coord_dir
        self.segmask_dir = segmask_dir
        self.num_slices = num_slices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row["series_id"]
        study_id = row["study_id"]
        
        image_tensor = torch.zeros((self.num_slices, 1, 512, 512), dtype=torch.float32)
        heatmap_tensor = torch.zeros((self.num_slices, 1, 512, 512), dtype=torch.float32)
        segmask_tensor = torch.zeros((self.num_slices, 1, 512, 512), dtype=torch.float32)
        coords = torch.zeros((self.num_slices, 2), dtype=torch.float32)
        labels = torch.full((self.num_slices, len(disease_cols)), -1, dtype=torch.long)

        for i in range(self.num_slices):
            img_path = os.path.join(self.image_dir, f"{study_id}_{series_id}_slice_{i:02d}.png")
            heatmap_path = os.path.join(self.heatmap_dir, f"{study_id}_{series_id}_slice_{i:02d}.png")
            coord_path = os.path.join(self.coord_dir, f"{study_id}_{series_id}_{i}.npy")
            seg_path = os.path.join(self.segmask_dir, f"{study_id}_{series_id}_slice_{i:02d}.png")

            if os.path.exists(seg_path):
                seg = self.load_image(seg_path)
                segmask_tensor[i] = seg

            if os.path.exists(img_path):
                img = self.load_image(img_path)
                image_tensor[i] = img

            if os.path.exists(heatmap_path):
                heatmap = self.load_image(heatmap_path)
                heatmap_tensor[i] = heatmap

            if os.path.exists(coord_path):
                coords[i] = torch.tensor(np.load(coord_path), dtype=torch.float32)

            for d, col in enumerate(disease_cols):
                if col in row and pd.notna(row[col]):
                    labels[i, d] = int(row[col])
        # ✅ 여기로 디버깅 이동 (반복문 바깥)
        if idx < 3:
            print(f"[DEBUG __getitem__] image_tensor.shape: {image_tensor.shape}")  # (25, 1, 512, 512)
            print(f"[DEBUG __getitem__] coords.shape: {coords.shape}")              # (25, 2)
            print(f"[DEBUG __getitem__] labels.shape: {labels.shape}")              # (25, 25)
            print(f"[DEBUG __getitem__] unique labels: {torch.unique(labels)}")        


        return {
            "image": image_tensor,
            "heatmap": heatmap_tensor,
            "segmask" : segmask_tensor,
            "coords": coords.view(-1),
            "label": labels
        }

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (512, 512))
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        return img