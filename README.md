# X-Ray Contraband Detection
Deep learning project for automated X-ray baggage inspection to detect weapons and tools using YOLOv8.

This repository contains the complete research pipeline for contraband detection on X-ray images. The project explores the effectiveness of YOLOv8 with Focal Loss for handling severe class imbalance in security screening datasets.

**Research language**: Russian (Colab notebooks are fully documented in Russian as scientific research papers)

## Research Links

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| **Root Pipeline** | Main pipeline: training, evaluation, visualization & conclusions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PNnd1Z3NGXygrD3EHdVNDNkdMYbm1CfI?usp=sharing) |
| **Data Exploration** | Dataset search, EDA, class analysis, YOLO format conversion | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1awF_W9cwKyAH5CEG5Zu9eFCqnEh0S0pV?usp=sharing) |
| **Model Experiments** | YOLOv8 training, Focal Loss modification, comparative analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mUdqMrityTx9U20FL49f-CqZ5egyWzi4?usp=sharing) |

---

## Problem Statement

Manual X-ray baggage inspection is:
- **Time-consuming** — operators must analyze thousands of images per shift
- **Prone to human error** — fatigue and loss of concentration reduce detection accuracy
- **Inefficient** — growing passenger and cargo flows require automation

**Goal**: Develop an automated detection system for prohibited items on X-ray images using deep learning.

---
## Dataset

This project uses a cleaned and curated version of the **PIDray (Prohibited Item Detection ray)** dataset — a large-scale X-ray benchmark introduced at ICCV 2021 for contraband detection in security screening.

### Dataset Summary

| Property | Value |
|----------|-------|
| **Source** | PIDray (Wang et al., ICCV 2021) |
| **Total images** | 47,674 |
| **Classes** | 12 (guns, knives, tools, handcuffs, etc.) |
| **Annotation format** | COCO JSON (bounding boxes) |
| **Splits** | Train (80%), Validation (10%), Test (10%) |

### Repository Contents

- `dataset_split.csv` — Full split table with image-to-set mapping
- `splits/` — File lists for train/val/test (`train_images.txt`, `val_images.txt`, `test_images.txt`)

### Full Dataset on Kaggle

The complete dataset (images + info) is available on Kaggle:

[![Kaggle](https://img.shields.io/badge/Kaggle-XRay%20Contraband%20Detection-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/BANan41k/xray-contraband-detection)

The complete yolo dataset is available on Kaggle:

[![Kaggle](https://img.shields.io/badge/Kaggle-XRay%20Contraband%20Detection-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)]([https://www.kaggle.com/datasets/BANan41k/xray-contraband-detection](https://www.kaggle.com/datasets/banan41k/xray-contraband-yolo))
