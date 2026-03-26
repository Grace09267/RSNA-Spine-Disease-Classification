import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from registration_model import SimpleRegNet, flow_sample, prepare_image_tensor, warp, resize_image_and_coords
import yaml

with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)

# --------------------------------------
# 1. Load Meta Data
# --------------------------------------
label_df = pd.read_csv(cfg["data"]["train_label_csv"])
desc_df = pd.read_csv(cfg["data"]["train_desc_csv"])

label_df['instance_number'] = label_df['instance_number'].astype(int)

# failed_cases.csv(608등의 해상도 차이때문에 못 돌아간 케이스)
RETRY_FAILED_ONLY = True
def load_failed_cases(cfg):
    failed_csv = cfg["output"]["failed_csv"]

    if not os.path.exists(failed_csv):
        return set()

    df = pd.read_csv(failed_csv)

    failed = set(
        zip(
            df.study_id.astype(str),
            df.series_id.astype(str)
        )
    )
    print(f"[INFO] retry targets: {len(failed)} series")

    return failed
# --------------------------------------
# 2. Anchor 후보 추출
# --------------------------------------
def extract_valid_anchors(desc_df, label_df):
    anchors = desc_df.merge(label_df[['study_id', 'series_id', 'instance_number']], on=['study_id', 'series_id'])
    return anchors.drop_duplicates()
# 모든 슬라이스
def extract_all_slice_anchors(desc_df):
    base_path = cfg["data"]["train_images_root"]
    anchors = []

    for _, row in desc_df.iterrows():
        study_id = row.study_id
        series_id = row.series_id

        dcm_dir = os.path.join(base_path, str(study_id), str(series_id))
        if not os.path.exists(dcm_dir):
            continue

        files = [f for f in os.listdir(dcm_dir) if f.endswith(".dcm")]

        for f in files:
            inst = int(f.replace(".dcm", ""))
            anchors.append({
                "study_id": study_id,
                "series_id": series_id,
                "instance_number": inst
            })

    return pd.DataFrame(anchors)

# --------------------------------------
# 3. DICOM 존재 확인
# --------------------------------------
def check_dicom_exists(df):
    base_path = cfg["data"]["train_images_root"]
    valids = []
    for _, row in df.iterrows():
        path = os.path.join(base_path, str(row.study_id), str(row.series_id))
        if os.path.exists(path):
            valids.append(row)
    return pd.DataFrame(valids)

# --------------------------------------
# 4. 슬라이스 불러오기
# --------------------------------------
TARGET_SIZE = 640

def load_slices(study_id, series_id):
    base_path = cfg["data"]["train_images_root"]
    dcm_dir = os.path.join(base_path, str(study_id), str(series_id))
    slices = []
    for f in os.listdir(dcm_dir):
        if not f.endswith(".dcm"):
            continue
        path = os.path.join(dcm_dir, f)
        try:
            dcm = pydicom.dcmread(path, force=True)
            if not hasattr(dcm,"ImagePositionPatient"):
                continue
            img = dcm.pixel_array.astype(np.float32)

            # -----------------------------
            # intensity normalize
            # -----------------------------
            img -= img.min()
            img /= (img.max() + 1e-6)  

            # -----------------------------
            # ⭐ RESOLUTION NORMALIZATION
            # -----------------------------
            h, w = img.shape
            if h != TARGET_SIZE or w != TARGET_SIZE:
                img = cv2.resize(
                    img,
                    (TARGET_SIZE, TARGET_SIZE),
                    interpolation=cv2.INTER_LINEAR
                )                      

            # ⭐ 진짜 slice 위치
            z_pos = float(dcm.ImagePositionPatient[2])
            inst = int(dcm.InstanceNumber)
            slices.append({
                "instance": inst,
                "z": z_pos,
                "image": img
            })
        except:
            continue

    # ⭐ 공간 기준 정렬
    slices = sorted(slices, key=lambda x: x["z"])
    return slices

# --------------------------------------
# 5. Hybrid Registration + Cycle Check
# 1) Anchor 좌표 기준
# 2) Forward flow 계산
# 3) Cycle consistency 체크
# 4) 실패 시 registration 재학습
# 5) Heatmap 저장
# --------------------------------------
def train_hybrid_registration(anchor_img, target_img, coords, model, steps = cfg["preprocess"]["registration_steps"], lam=1.0):
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
# 6. 전체 전파 루프
# --------------------------------------
# register step(GPU SAFE)
def register_pair(model, img_a, img_b, coords):
    with torch.no_grad():
        a = prepare_image_tensor(img_a)
        b = prepare_image_tensor(img_b)
    with torch.no_grad():
        flow = model(a, b)
        disp = flow_sample(flow, coords)
    return (coords + disp).detach()

# Online adaptation step
def online_adapt(model, img_a, img_b, coords, steps=2):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    a = prepare_image_tensor(img_a)
    b = prepare_image_tensor(img_b)
    coords = coords.detach()
    for _ in range(steps):
        flow = model(a, b)
        warped = warp(b, flow)
        loss_img = F.mse_loss(warped, a)
        disp = flow_sample(flow, coords)
        pred = coords + disp
        loss_pts = F.mse_loss(pred, coords)
        loss = loss_img + 0.5 * loss_pts
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

# DICOM Info 기준으로 전파
def propagate_labels(study_id, series_id,
                     anchor_instance,
                     coords_gt,
                     model,
                     slices):
    results = []
    cycle_threshold = cfg["preprocess"]["cycle_threshold"]
    # anchor index 찾기
    anchor_idx = next(
        i for i, s in enumerate(slices)
        if s["instance"] == anchor_instance
    )
   
    # 좌표 슬라이스별 기록용
    coords = coords_gt.clone().detach()
    coords_per_slice = [None] * len(slices)
    coords_per_slice[anchor_idx] = coords.clone() # anchor는 GT
    
    # ===============================
    # 1️⃣ GT → RIGHT END
    # ===============================
    for i in range(anchor_idx + 1, len(slices)):
        coords = register_pair(
            model,
            slices[i-1]["image"],
            slices[i]["image"],
            coords
        )
        coords_per_slice[i] = coords.clone()
        if i % 5 == 0:   # GPU SAFE adaptation
            online_adapt(
                model,
                slices[i-1]["image"],
                slices[i]["image"],
                coords,
                steps=2
            )
    # ===============================
    # 2️⃣ RIGHT → LEFT END
    # ===============================
    for i in reversed(range(1, len(slices))):
        coords = register_pair(
            model,
            slices[i]["image"],
            slices[i-1]["image"],
            coords
        )
        coords_per_slice[i-1] = coords.clone()
    # ===============================
    # 3️⃣ LEFT → GT (cycle close)
    # ===============================
    for i in range(1, anchor_idx + 1):
        coords = register_pair(
            model,
            slices[i-1]["image"],
            slices[i]["image"],
            coords
        )
    coords_cycle = coords

    # ===============================
    # 4️⃣ None 값 처리: 선형보간 및 fallback
    # ===============================
    for i in range(len(coords_per_slice)):
        if coords_per_slice[i] is None:
            coords_per_slice[i] = coords_per_slice[anchor_idx].clone()

    # ===============================
    # 4️⃣ Cycle Error
    # ===============================
    cycle_error = torch.norm(coords_cycle - coords_gt, dim=1).mean().item()
    is_valid = cycle_error < cycle_threshold
    for i, slice in enumerate(slices):
        coords_slice = coords_per_slice[i]
        results.append({
            "study_id": study_id,
            "series_id": series_id,
            "anchor_instance": anchor_instance,
            "instance_number": slice["instance"],
            "num_points":coords_slice.shape[0],
            "x_coords": coords_slice[:,0].cpu().numpy().tolist(),
            "y_coords": coords_slice[:,1].cpu().numpy().tolist(),
            "cycle_error": cycle_error,
            "valid": int(is_valid)
        })
    torch.cuda.empty_cache()
    return results


# --------------------------------------
# 7. 전체 자동 실행 + 실패 케이스 로깅
# --------------------------------------
def run_propagation():
    # failed_cases--------------------------------
    failed_targets = None
    if RETRY_FAILED_ONLY:
        failed_targets = load_failed_cases(cfg)
    #----------------------------------------------
    all_results = []
    failed_cases = []
    anchors = extract_valid_anchors(desc_df, label_df)
    valid_anchors = check_dicom_exists(anchors)
    #model = SimpleRegNet().cuda()
    #model.eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # for i, row in tqdm(valid_anchors.head(3).iterrows(), total=3):
    for i, row in tqdm(valid_anchors.iterrows(), total=len(valid_anchors)):
        # -------------------------
        # retry mode filtering(failed_cases)
        model = SimpleRegNet().cuda()
        model.eval()
        if RETRY_FAILED_ONLY:
            key = (str(row.study_id), str(row.series_id))
            if key not in failed_targets:
                continue   
        #--------------------------------------
        try:
            slices = load_slices(row.study_id, row.series_id)
            #anchor_img = next(s["image"] for s in slices if s["instance"] == row.instance_number)
            # anchor slice 안전하게 찾기
            anchor_slice = next((s for s in slices if s["instance"] == int(row.instance_number)), None)
            if anchor_slice is None:
                print(f"[WARNING] Anchor slice not found: study {row.study_id}, series {row.series_id}, instance {row.instance_number}")
                failed_cases.append({
                    "study_id": row.study_id,
                    "series_id": row.series_id,
                    "error": "Anchor slice not found"
                })
                continue  # 다음 row로 넘어감
            anchor_img = anchor_slice["image"]
                        
            coords_df = label_df[(label_df.study_id == row.study_id) &
                                 (label_df.series_id == row.series_id) &
                                 (label_df.instance_number == row.instance_number)]
            
            if coords_df.empty:
                print(f"[WARNING] No coordinates found in label.csv for study {row.study_id}, series {row.series_id}, instance {row.instance_number}")
                failed_cases.append({
                    "study_id": row.study_id,
                    "series_id": row.series_id,
                    "error": "No coordinates in label.csv"
                })
                continue
            coords = torch.tensor(coords_df[["x", "y"]].values, dtype=torch.float32).cuda()
            
            # anchor 기준 hybrid registration 학습
            model = train_hybrid_registration(anchor_img, anchor_img, coords, model, steps=cfg["preprocess"].get("registration_steps", 10))
            
            results = propagate_labels(row.study_id, row.series_id, row.instance_number, coords, model, slices)
            all_results.extend(results)
        except Exception as e:
            failed_cases.append({"study_id": row.study_id, "series_id": row.series_id, "error": str(e)})

    #os.makedirs(os.path.dirname(cfg["output"]["propagated_csv"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output"]["failed_csv"]), exist_ok=True)
    #pd.DataFrame(all_results).to_csv(cfg["output"]["propagated_csv"], index=False)
    pd.DataFrame(failed_cases).to_csv(cfg["output"]["failed_csv"], index=False)
    print(f"총 실패 케이스: {len(failed_cases)}")
    # failed_cases
    prop_path = cfg["output"]["propagated_csv"]
    os.makedirs(os.path.dirname(prop_path), exist_ok=True)
    new_df = pd.DataFrame(all_results)
    # ----------------------------------
    # append mode
    # ----------------------------------
    if os.path.exists(prop_path):
        old_df = pd.read_csv(prop_path)

        print(f"[INFO] existing rows: {len(old_df)}")
        print(f"[INFO] new rows: {len(new_df)}")

        merged_df = pd.concat([old_df, new_df], ignore_index=True)

        # ⭐ 중복 제거 (VERY IMPORTANT)
        merged_df = merged_df.drop_duplicates(
            subset=[
                "study_id",
                "series_id",
                "instance_number",
                "anchor_instance"
            ],
            keep="last"
        )

    else:
        merged_df = new_df

    merged_df.to_csv(prop_path, index=False)

    print(f"[INFO] total saved rows: {len(merged_df)}")

# 실행
if __name__ == "__main__":
    run_propagation()