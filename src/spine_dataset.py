import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class SpineDataset(Dataset):

    def __init__(self, root_dir, indices, downsample=True):

        self.images = np.memmap(
            os.path.join(root_dir,"images.npy"),
            dtype=np.float16,
            mode="r",
            shape=(351783,9,224,224)
        )

        self.labels = np.memmap(
            os.path.join(root_dir,"labels.npy"),
            dtype=np.int64,
            mode="r",
            shape=(351783,5)
        )

        self.coords = np.memmap(
            os.path.join(root_dir,"coords.npy"),
            dtype=np.float32,
            mode="r",
            shape=(351783,2)
        )

        self.levels = np.memmap(
            os.path.join(root_dir,"levels.npy"),
            dtype=np.int64,
            mode="r",
            shape=(351783,)
        )

        self.indices = []

        for i in range(len(self.labels)):

            label = self.labels[i]  # (5,)

            if not downsample:
                self.indices.append(i)
                continue

            # 하나라도 0이 아닌 label이 있으면 무조건 포함
            if (label > 0).any():
                self.indices.append(i)

            else:
                # 전부 0인 경우 → 1/3 확률로만 사용
                if np.random.rand() < 0.33:
                    self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        img = torch.from_numpy(self.images[real_idx]).float()
        label = torch.from_numpy(self.labels[real_idx]).long()
        if (label < 0).any():
            label = torch.clamp(label, min=0)
        coord = torch.from_numpy(self.coords[real_idx]).float()
        level = torch.tensor(self.levels[real_idx]).long()

        # CNN 입력용 channel 확장
        img = img.unsqueeze(1)      # [9,1,224,224]
        img = img.repeat(1,3,1,1)   # [9,3,224,224]

        return {
            "image": img,
            "label": label,
            "coord": coord,
            "level": level
        }


def build_datasets(root_dir, val_ratio=0.1):

    labels = np.memmap(
    os.path.join(root_dir,"labels.npy"),
    dtype=np.int64,
    mode="r",
    shape=(351783,5))

    all_idx = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=val_ratio,
        random_state=42,
        shuffle=True
    )

    train_dataset = SpineDataset(root_dir, train_idx, downsample=True)
    val_dataset = SpineDataset(root_dir, val_idx, downsample=True)

    print("=================================")
    print("Total samples:", len(all_idx))
    print("Train samples:", len(train_idx))
    print("Val samples:", len(val_idx))
    print("=================================")

    return train_dataset, val_dataset