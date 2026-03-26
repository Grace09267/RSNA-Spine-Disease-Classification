import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import random

def seed_everything(seed):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

class SpineDataset(Dataset):

    def __init__(self, root_dir, indices, downsample=True):
        np.random.seed(42) # 시드 고정 한번더
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
        pos_idx = []
        neg_idx = []

        for i in indices:

            label = self.labels[i]  # (5,)

            # 하나라도 0이 아닌 label이 있으면 무조건 포함
            if (label > 0).any():
                pos_idx.append(i)
            else:
                neg_idx.append(i)

        # 고정된 샘플링
        rng = np.random.RandomState(42)
        neg_sample = rng.choice(neg_idx, size=len(neg_idx)//5, replace=False)

        self.indices = pos_idx + list(neg_sample)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        img = torch.from_numpy(self.images[real_idx].copy()).float()
        label = torch.from_numpy(self.labels[real_idx].copy()).long()
        if (label < 0).any():
            label = torch.clamp(label, min=0)
        coord = torch.from_numpy(self.coords[real_idx].copy()).float()
        level = torch.tensor(self.levels[real_idx]).long()

        # CNN 입력용 channel 확장
        img = img.unsqueeze(1)      # [9,1,224,224]
        #img = img.repeat(1,3,1,1)   # [9,3,224,224]

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
    val_dataset = SpineDataset(root_dir, val_idx, downsample=False)

    print("=================================")
    print("Total samples:", len(all_idx))
    print("Train samples:", len(train_idx))
    print("Val samples:", len(val_idx))
    print("=================================")

    return train_dataset, val_dataset