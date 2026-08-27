# X-Ray Contraband Detection
Deep learning project for automated X-ray baggage inspection to detect weapons and tools using YOLOv8.

This repository contains the complete research pipeline for contraband detection on X-ray images. The project explores the effectiveness of YOLOv8 with Focal Loss for handling severe class imbalance in security screening datasets.

**Research language**: Russian (Colab notebooks are fully documented in Russian as scientific research papers)

## Research Links

| Notebook | Description | Open in Colab |
|----------|-------------|---------------|
| **Root Pipeline** | Main pipeline: training, evaluation, visualization & conclusions | [![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg](https://colab.research.google.com/drive/1PNnd1Z3NGXygrD3EHdVNDNkdMYbm1CfI?usp=sharing))](https://colab.research.google.com/drive/1PNnd1Z3NGXygrD3EHdVNDNkdMYbm1CfI?usp=sharing) |
| **Data Exploration** | Dataset search, EDA, class analysis, YOLO format conversion | [![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg](https://colab.research.google.com/drive/1awF_W9cwKyAH5CEG5Zu9eFCqnEh0S0pV?usp=sharing))](https://colab.research.google.com/drive/1awF_W9cwKyAH5CEG5Zu9eFCqnEh0S0pV?usp=sharing) |
| **Model Experiments** | YOLOv8 training, Focal Loss modification, comparative analysis | [![Open In Colab]([https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mUdqMrityTx9U20FL49f-CqZ5egyWzi4?usp=sharing](https://colab.research.google.com/drive/1mUdqMrityTx9U20FL49f-CqZ5egyWzi4?usp=sharing)) |

---

## Problem Statement

Manual X-ray baggage inspection is:
- **Time-consuming** — operators must analyze thousands of images per shift
- **Prone to human error** — fatigue and loss of concentration reduce detection accuracy
- **Inefficient** — growing passenger and cargo flows require automation

**Goal**: Develop an automated detection system for prohibited items on X-ray images using deep learning.

---
