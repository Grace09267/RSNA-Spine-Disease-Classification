RSNA Spine Disease Classification

📌 Project Overview
This project aims to develop an automated disease classification pipeline for lumbar spine MRI using deep learning.

The pipeline integrates:
Registration-based label propagation
Pseudo segmentation generation
Attention-based multi-label classification
This project was developed using the RSNA 2024 Lumbar Spine MRI dataset.

📌 Motivation
Manual annotation of spine MRI is time-consuming and requires expert radiologists.

To reduce annotation cost and improve scalability, this project proposes:
Propagating labels using image registration
Generating pseudo segmentation masks
Leveraging spatial attention for disease classification

📌 Pipeline Architecture
MRI DICOM Slices
        ↓
Registration-based Label Propagation
        ↓
Pseudo Segmentation Mask Generation
        ↓
Attention-based Classification Model
        ↓
Multi-disease Prediction (25 Labels)

📌 Dataset
Dataset: RSNA 2024 Lumbar Spine MRI

Input:
DICOM slices
Propagated anatomical coordinates
Heatmap representations

Output:
Multi-label classification for 25 disease categories
※ Dataset is not included due to licensing restrictions.

📌 Methodology
1️⃣ Registration-based Label Propagation
CNN flow-based registration model
Self-supervised optimization
Cycle error used for quality validation

2️⃣ Pseudo Segmentation Generation
Heatmap-based mask generation
UNet-based segmentation structure
Used as auxiliary spatial feature

3️⃣ Attention-based Classification
Fusion model using:
Image features
Heatmap features
Segmentation features
Coordinate-based spatial attention

📌 Progress
1.pre_process_all.py


