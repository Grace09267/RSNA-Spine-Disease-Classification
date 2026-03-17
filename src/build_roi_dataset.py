import os
import ast
import torch
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
import cv2

#############################################
# CONFIG
#############################################
CFG = dict(
    data_root="/mnt/c/Users/lhe09/projects/rsna2024_spine/Data/train_images",
    propagation_csv="/home/lhe09/projects/rsna_spine/outputs/labels/propagated_labels.csv",
    label_csv="/mnt/c/Users/lhe09/projects/rsna2024_spine/Data/train.csv",
    save_dir="/home/lhe09/projects/rsna_spine/outputs/roi_dataset",

    roi_mm=70,
    num_slices=9,
    img_size=224
)

os.makedirs(CFG["save_dir"], exist_ok=True)

DISC_LEVELS = [
    "L1/L2",
    "L2/L3",
    "L3/L4",
    "L4/L5",
    "L5/S1"
]

LABEL_MAP = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2
}

#############################################
# DICOM LOAD + SORT (CRITICAL)
#############################################
def load_series(study_id, series_id):

    dcm_dir = os.path.join(
        CFG["data_root"],
        str(study_id),
        str(series_id)
    )

    slices = []

    for f in os.listdir(dcm_dir):
        if not f.endswith(".dcm"):
            continue

        path = os.path.join(dcm_dir, f)
        try:
            dcm = pydicom.dcmread(path)
        except:
            continue

        if not hasattr(dcm, "ImagePositionPatient"):
            continue

        img = dcm.pixel_array.astype(np.float32)
        img -= img.min()
        img /= (img.max() + 1e-6)

        slices.append({
            "instance": int(dcm.InstanceNumber),
            "z": float(dcm.ImagePositionPatient[2]),
            "img": img,
            "spacing": list(map(float, dcm.PixelSpacing))
        })

    slices = sorted(slices, key=lambda x: x["z"])
    return slices

# UTILS
#############################################
def mm_to_pixel(mm, spacing):
    return int(mm/spacing[0]), int(mm/spacing[1])


#############################################
def crop_roi(img, center, roi_px):

    h, w = img.shape
    cx, cy = int(center[0]), int(center[1])
    rx, ry = roi_px

    x1 = max(cx-rx//2, 0)
    x2 = min(cx+rx//2, w)
    y1 = max(cy-ry//2, 0)
    y2 = min(cy+ry//2, h)

    roi = img[y1:y2, x1:x2]

    roi = cv2.resize(
        roi,
        (CFG["img_size"], CFG["img_size"]),
        interpolation=cv2.INTER_LINEAR
    )
    return roi


#############################################
def build_slice_indices(center_idx, total):

    half = CFG["num_slices"]//2
    ids = []

    for i in range(center_idx-half, center_idx+half+1):
        i = max(0, min(total-1, i))
        ids.append(i)

    return ids


#############################################
# LABEL LOAD
#############################################
def load_labels():

    df = pd.read_csv(CFG["label_csv"])

    label_map = {}

    for _, r in df.iterrows():

        study_id = r.study_id

        label_map[(study_id,"L1/L2")] = [
            LABEL_MAP.get(r["spinal_canal_stenosis_l1_l2"], -1),
            LABEL_MAP.get(r["left_neural_foraminal_narrowing_l1_l2"], -1),
            LABEL_MAP.get(r["right_neural_foraminal_narrowing_l1_l2"], -1),
            LABEL_MAP.get(r["left_subarticular_stenosis_l1_l2"], -1),
            LABEL_MAP.get(r["right_subarticular_stenosis_l1_l2"], -1),
        ]

        label_map[(study_id,"L2/L3")] = [
            LABEL_MAP.get(r["spinal_canal_stenosis_l2_l3"], -1),
            LABEL_MAP.get(r["left_neural_foraminal_narrowing_l2_l3"], -1),
            LABEL_MAP.get(r["right_neural_foraminal_narrowing_l2_l3"], -1),
            LABEL_MAP.get(r["left_subarticular_stenosis_l2_l3"], -1),
            LABEL_MAP.get(r["right_subarticular_stenosis_l2_l3"], -1),
        ]

        label_map[(study_id,"L3/L4")] = [
            LABEL_MAP.get(r["spinal_canal_stenosis_l3_l4"], -1),
            LABEL_MAP.get(r["left_neural_foraminal_narrowing_l3_l4"], -1),
            LABEL_MAP.get(r["right_neural_foraminal_narrowing_l3_l4"], -1),
            LABEL_MAP.get(r["left_subarticular_stenosis_l3_l4"], -1),
            LABEL_MAP.get(r["right_subarticular_stenosis_l3_l4"], -1),
        ]

        label_map[(study_id,"L4/L5")] = [
            LABEL_MAP.get(r["spinal_canal_stenosis_l4_l5"], -1),
            LABEL_MAP.get(r["left_neural_foraminal_narrowing_l4_l5"], -1),
            LABEL_MAP.get(r["right_neural_foraminal_narrowing_l4_l5"], -1),
            LABEL_MAP.get(r["left_subarticular_stenosis_l4_l5"], -1),
            LABEL_MAP.get(r["right_subarticular_stenosis_l4_l5"], -1),
        ]

        label_map[(study_id,"L5/S1")] = [
            LABEL_MAP.get(r["spinal_canal_stenosis_l5_s1"], -1),
            LABEL_MAP.get(r["left_neural_foraminal_narrowing_l5_s1"], -1),
            LABEL_MAP.get(r["right_neural_foraminal_narrowing_l5_s1"], -1),
            LABEL_MAP.get(r["left_subarticular_stenosis_l5_s1"], -1),
            LABEL_MAP.get(r["right_subarticular_stenosis_l5_s1"], -1),
        ]

    return label_map


#############################################
# MAIN
#############################################
def preprocess():

    prop_df = pd.read_csv(CFG["propagation_csv"])
    # propagation 실패 제거, cycle_error > 0.2 이상 제거
    prop_df = prop_df[
        (prop_df.valid == 1) &
        (prop_df.cycle_error < 0.2)
    ]
    # 같은 slice 중복제거
    prop_df = prop_df.drop_duplicates(
        subset=["study_id","series_id","instance_number"]
    )

    label_map = load_labels()

    sample_id = 0
# series 단위로 그룹
    groups = prop_df.groupby(["study_id", "series_id"])
    for (study_id, series_id), gdf in tqdm(groups):

        try:
            slices = load_series(study_id, series_id)
        except:
            continue

        if len(slices) == 0:
            continue

        inst_map = {
            s["instance"]: i
            for i, s in enumerate(slices)
        }

        #############################################
        # slice loop
        #############################################

        for _, row in gdf.iterrows():
            inst = row.instance_number

            if inst not in inst_map:
                continue

            center_idx = inst_map[inst]
            slice_ids = build_slice_indices(
                center_idx,
                len(slices)
            )
            coords_x = ast.literal_eval(row.x_coords)
            coords_y = ast.literal_eval(row.y_coords)
            spacing = slices[center_idx]["spacing"]
            roi_px = mm_to_pixel(
                CFG["roi_mm"],
                spacing
            )

            #################################
            # ⭐ DISC별 SAMPLE 생성
            #################################
            disc_count = min(len(coords_x), len(DISC_LEVELS))
            for disc_i in range(disc_count):
                level = DISC_LEVELS[disc_i]
                label = label_map.get(
                    (study_id, level),
                    [-1,-1,-1,-1,-1]
                )
                label = torch.tensor(label).long()

                stack = []

                # 좌표가 짤리지 않게 좌표를 image boundary 안으로 강제로 넣음
                cx = np.clip(coords_x[disc_i], 0, slices[center_idx]["img"].shape[1]-1)
                cy = np.clip(coords_y[disc_i], 0, slices[center_idx]["img"].shape[0]-1)

                center = (cx, cy)

                for sid in slice_ids:

                    img = slices[sid]["img"]

                    roi = crop_roi(img, center, roi_px)
                    stack.append(roi)

                stack = torch.tensor(np.stack(stack)).half() # half()는 float()보다 저장 dtype 줄이는 방법
                coord = torch.tensor(center).float()

                save_dict = dict(
                    image=stack,              # [9,224,224]
                    label=label,              # classification label[5]
                    level=level,              # L1/L2 ~ L5/S1
                    coord=coord,              # propagated center
                    study_id=study_id,
                    series_id=series_id,
                    instance_number=inst,

                    cycle_error=row.cycle_error
                )

                torch.save(
                    save_dict,
                    os.path.join(
                        CFG["save_dir"],
                        f"{sample_id}.pt"
                    )
                )

                sample_id += 1


if __name__ == "__main__":
    preprocess()