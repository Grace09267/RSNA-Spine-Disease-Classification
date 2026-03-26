import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

# --------------------------
# SimpleRegNet: CNN 기반 Registration 모델
# --------------------------
class SimpleRegNet(nn.Module):
    def __init__(self):
        super(SimpleRegNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 2, 3, padding=1)  # flow 출력 (dx, dy)
        )

    def forward(self, source, target):
        source = source.contiguous()
        target = target.contiguous()
        x = torch.cat([source, target], dim=1)  # (B, 2, H, W)
        x = self.encoder(x)
        flow = self.decoder(x)
        return flow

# --------------------------
# flow_sample: flow 필드에서 특정 좌표에 대한 벡터 추출
# --------------------------
def flow_sample(flow, coords):
    N = coords.shape[0]
    H, W = flow.shape[2], flow.shape[3]

    norm_coords = coords.clone()
    norm_coords[:, 0] = (norm_coords[:, 0] / (W - 1)) * 2 - 1
    norm_coords[:, 1] = (norm_coords[:, 1] / (H - 1)) * 2 - 1
    grid = norm_coords.view(1, N, 1, 2)

    sampled = F.grid_sample(flow, grid, align_corners=True, mode='bilinear')
    sampled = sampled.squeeze(-1).squeeze(0).permute(1, 0)  # (2, N) -> (N, 2)
    return sampled

# --------------------------
# warp: flow를 이용해 이미지를 변형 (warping)
# --------------------------
_grid_cache = {}
def warp(image, flow):
    B, C, H, W = image.shape
    device = image.device
    key = (H, W, device)
    if key not in _grid_cache:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        # grid shape=(H, W, 2)
        grid = torch.stack((grid_x, grid_y), 2).float()
        _grid_cache[key] = grid

#    grid_y, grid_x = torch.meshgrid(
#        torch.arange(0, H, device=image.device),
#        torch.arange(0, W, device=image.device), indexing='ij')
#    grid = torch.stack((grid_x, grid_y), 2).float()
#    grid = grid.unsqueeze(0) # (1, H, W, 2)
    
    grid = _grid_cache[key] # (H, W, 2)

    # flow 보간 (anchor와 동일 크기로 맞춤)
    flow = F.interpolate(flow, size=(H, W), mode='bilinear', align_corners=True)
    
    grid = grid.unsqueeze(0) + flow.permute(0, 2, 3, 1)  # (1, H, W, 2)
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

    warped = F.grid_sample(image, grid, align_corners=True, mode='bilinear')
    return warped

# --------------------------
# prepare_image_tensor: numpy 이미지를 (1, 1, H, W) 텐서로 변환 후 GPU에 업로드
# --------------------------425개의 실패 케이스가 생김(압축건의 문제?로 5D이 생겨서 4D로 바뀌기 위해 아래 코드로 수정)
# def prepare_image_tensor(np_image, device="cuda"):
#     norm = (np_image - np_image.min()) / (np_image.max() - np_image.min() + 1e-8)
#     tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     return tensor.to(device)
def prepare_image_tensor(image, device="cuda"):
    image = np.array(image, dtype=np.float32)
    # ✅ squeeze unexpected dims
    image = np.squeeze(image)
#    # normalize
#    image = (image - image.min()) / (image.max() - image.min() + 1e-6)
#    tensor = torch.from_numpy(image)

    # 반드시 2D 보장
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D after squeeze, got {image.shape}")
#    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    
    # robust normalization(percentile로 구조유지하기 위하여)
    p1, p99 = np.percentile(image, (1, 99))
    if abs(p99 - p1) < 1e-6:
        image = np.zeros_like(image, dtype=np.float32)
    else:
        image = np.clip(image, p1, p99)
        image = (image -p1) / (p99 - p1)
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    return tensor.to(device)
#    return tensor.cuda()


# source와 target의 이미지 크기를 맞추기 위해, 좌표와 이미지 모두 resize 진행함함
def resize_image_and_coords(image, coords, resize_size):
    h_orig, w_orig = image.shape
    w_new, h_new = resize_size

    resized_image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

    scale_x = w_new / w_orig
    scale_y = h_new / h_orig
    coords_scaled = coords.copy()
    coords_scaled[:, 0] *= scale_x
    coords_scaled[:, 1] *= scale_y

    return resized_image, coords_scaled