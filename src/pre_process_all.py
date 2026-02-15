import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from model import SimpleRegNet, flow_sample, prepare_image_tensor, warp, resize_image_and_coords

# --------------------------------------
# 1. Load Meta Data
# --------------------------------------
label_df = pd.read_csv("rsna_2024_spine/train_label_coordinates.csv")
desc_df = pd.read_csv("rsna_2024_spine/train_series_descriptions.csv")

# --------------------------------------
# 2. Anchor 후보 추출
# --------------------------------------
def extract_valid_anchors(desc_df, label_df):
    anchors = desc_df.merge(label_df[['study_id', 'series_id', 'instance_number']], on=['study_id', 'series_id'])
    return anchors.drop_duplicates()

# --------------------------------------
# 3. DICOM 존재 확인
# --------------------------------------
def check_dicom_exists(df, base_path="rsna_2024_spine/train_images"):
    valids = []
    for _, row in df.iterrows():
        path = os.path.join(base_path, str(row.study_id), str(row.series_id))
        if os.path.exists(path):
            valids.append(row)
    return pd.DataFrame(valids)

# --------------------------------------
# 4. 슬라이스 불러오기
# --------------------------------------
def load_slices(study_id, series_id, base_path="rsna_2024_spine/train_images"):
    dcm_dir = os.path.join(base_path, str(study_id), str(series_id))
    files = sorted([f for f in os.listdir(dcm_dir) if f.endswith(".dcm")])
    slices = []
    for f in files:
        path = os.path.join(dcm_dir, f)
        dcm = pydicom.dcmread(path)
        inst = int(dcm.InstanceNumber)
        slices.append((inst, dcm.pixel_array))
    return sorted(slices, key=lambda x: x[0])

# --------------------------------------
# 5. Hybrid Registration + Cycle Check
# --------------------------------------
def train_hybrid_registration(anchor_img, target_img, coords, model, steps=300, lam=1.0):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    anchor_tensor = prepare_image_tensor(anchor_img)
    target_tensor = prepare_image_tensor(target_img)
    coords = coords.cuda()
    for step in range(steps):
        flow = model(anchor_tensor, target_tensor)
        warped_target = warp(target_tensor, flow)
        loss_img = F.mse_loss(warped_target, anchor_tensor)
        pred_disp = flow_sample(flow, coords)
        pred_coords = coords + pred_disp
        loss_pts = F.mse_loss(pred_coords, coords)
        loss = loss_img + lam * loss_pts
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

# --------------------------------------
# 히트맵 저장 함수 (npy + 이미지)
# --------------------------------------
def save_heatmap(image, coords, save_path_png, save_path_npy):
    heat = np.zeros_like(image, dtype=np.float32)
    for x, y in coords:
        if 0 <= int(y) < heat.shape[0] and 0 <= int(x) < heat.shape[1]:
            heat[int(y), int(x)] += 1.0
    heat = cv2.GaussianBlur(heat, (11, 11), 0)
    np.save(save_path_npy, heat)
    heat_img = (heat / heat.max() * 255).astype(np.uint8) if heat.max() > 0 else heat
    plt.imsave(save_path_png, heat_img, cmap='jet')

# --------------------------------------
# 6. 전체 전파 루프
# --------------------------------------
def propagate_labels(study_id, series_id, anchor_instance, coords, model, slices, threshold=0.2, heatmap_dir="heatmaps"):
    results = []
    anchor_img = [img for inst, img in slices if inst == anchor_instance][0]
    anchor_img, coords_resized = resize_image_and_coords(anchor_img, coords.cpu().numpy(), size=(512, 512))
    anchor_tensor = prepare_image_tensor(anchor_img)
    coords = torch.tensor(coords_resized, dtype=torch.float32).cuda()
    os.makedirs(heatmap_dir, exist_ok=True)

    for inst_num, target_img in slices:
        if inst_num == anchor_instance:
            continue
        target_img_resized, _ = resize_image_and_coords(target_img, coords_resized, size=(512, 512))
        target_tensor = prepare_image_tensor(target_img_resized)

        flow_ab = model(anchor_tensor, target_tensor)
        coords_in_target = coords + flow_sample(flow_ab, coords)
        flow_ba = model(target_tensor, anchor_tensor)
        coords_back = coords_in_target + flow_sample(flow_ba, coords_in_target)
        cycle_error = (coords_back - coords).norm(dim=1)
        mean_error = cycle_error.mean().item()

        if mean_error > threshold:
            model = train_hybrid_registration(anchor_img, target_img_resized, coords, model)
            flow_ab = model(anchor_tensor, target_tensor)
            coords_in_target = coords + flow_sample(flow_ab, coords)

        np_coords = coords_in_target.detach().cpu().numpy()
        base_filename = f"{study_id}_{series_id}_{inst_num}"
        save_path_png = os.path.join(heatmap_dir, base_filename + ".png")
        save_path_npy = os.path.join(heatmap_dir, base_filename + ".npy")
        save_heatmap(target_img_resized, np_coords, save_path_png, save_path_npy)

        for i, pt in enumerate(coords_in_target):
            results.append({
                "study_id": study_id,
                "series_id": series_id,
                "instance_number": inst_num,
                "x": pt[0].item(),
                "y": pt[1].item(),
                "level": i + 1
            })
    return results

# --------------------------------------
# 7. 전체 자동 실행 + 실패 케이스 로깅
# --------------------------------------
def run_propagation():
    all_results = []
    failed_cases = []
    anchors = extract_valid_anchors(desc_df, label_df)
    valid_anchors = check_dicom_exists(anchors)
    model = SimpleRegNet().cuda()

    for i, row in tqdm(valid_anchors.iterrows(), total=len(valid_anchors)):
        try:
            slices = load_slices(row.study_id, row.series_id)
            anchor_img = [img for inst, img in slices if inst == row.instance_number][0]
            coords_df = label_df[(label_df.study_id == row.study_id) &
                                 (label_df.series_id == row.series_id) &
                                 (label_df.instance_number == row.instance_number)]
            coords = torch.tensor(coords_df[["x", "y"]].values, dtype=torch.float32).cuda()
            results = propagate_labels(row.study_id, row.series_id, row.instance_number, coords, model, slices)
            all_results.extend(results)
        except Exception as e:
            failed_cases.append({"study_id": row.study_id, "series_id": row.series_id, "error": str(e)})

    pd.DataFrame(all_results).to_csv("propagated_labels.csv", index=False)
    pd.DataFrame(failed_cases).to_csv("failed_cases.csv", index=False)
    print(f"총 실패 케이스: {len(failed_cases)}")

# 실행
run_propagation()

# 전처리 결과 시각화 예시 함수

def visualize_random_heatmap(n=5, heatmap_dir="heatmaps"):
    import random
    npy_files = [f for f in os.listdir(heatmap_dir) if f.endswith(".npy")]
    sampled = random.sample(npy_files, n)
    for file in sampled:
        npy_path = os.path.join(heatmap_dir, file)
        heatmap = np.load(npy_path)
        plt.imshow(heatmap, cmap="hot")
        plt.title(file)
        plt.colorbar()
        plt.show()
