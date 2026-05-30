# AI Deforestation Monitoring System: Automated Land Cover Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-orange?style=flat)](https://gradio.app/)

An end-to-end Computer Vision and Deep Learning pipeline designed to transform raw satellite pixel data from Earth Observation (EO) systems into actionable environmental intelligence to combat global deforestation. This project systematically builds, optimizes, and evaluates three distinct Convolutional Neural Network (CNN) architectures to classify Sentinel-2 satellite imagery into 10 specialized land cover categories.

---

## 📌 Project Overview
Rapid urbanization, industrial expansion, and aggressive deforestation present severe ecological crises. While modern satellites like the European Space Agency’s Sentinel-2 offer massive streams of high-resolution data, manual interpretation is impossible at scale. 

This project bridges that gap by deploying deep learning to automatically map spatial hierarchies and accurately classify land cover types despite high intra-class variance and inter-class spectral similarities (e.g., distinguishing between visually identical 'River' and 'SeaLake' tiles).

### 🗂️ Target Classes (10)
The models are trained on the **EuroSAT RGB dataset** (27,000 balanced images) to categorize landscapes into:
* `Annual Crop` / `Permanent Crop`
* `Forest` / `Herbaceous Vegetation` / `Pasture`
* `Highway` / `Industrial` / `Residential`
* `River` / `SeaLake`

---

## 🚀 Key Features
* **Progressive Architecture Scaling:** Evaluates custom scratch-built architectures against advanced pre-trained models via Transfer Learning.
* **Rotationally Invariant Data Augmentation:** Implements data transformations tailored for satellite perspectives (random horizontal/vertical flips, 90-degree rotations).
* **Web UI Deployment:** Integrates a local dash-style web app for real-time model evaluation, image uploading, and confidence mapping.

---

## 🧬 Architectural Evolution & Methodology

The project explores three development phases to optimize structural feature extraction:

### 1. Model 1: Baseline CNN (Custom Scratch-Built)
* **Inspiration:** AlexNet architecture.
* **Setup:** 5 Convolutional layers (using Large $11\times11$ down to $3\times3$ filters) with Max Pooling followed by 3 Fully Connected layers.
* **Regularization:** Standard 50% Dropout. Evaluates the raw capacity of networks initialized without advanced stabilizers.

### 2. Model 2: Improved CNN (Optimized Scratch-Built)
* **Setup:** Replicates Model 1 structure but natively integrates 2D Batch Normalization after every single Convolutional layer and 1D Batch Normalization inside the classifier head.
* **Enhancements:** Leverages Adaptive Average Pooling to fix spatial outputs and introduces rotation-based data augmentations to minimize overfitting and internal covariate shift.

### 3. Model 3: Fine-Tuned ResNet-50 (Transfer Learning)
* **Setup:** Leverages deep residual skip connections initialized with pre-trained ImageNet weights.
* **Strategy:** Early spatial feature extractors (Layers 1-3) are frozen to keep lower-level visual primitives, while Layer 4 is completely unfrozen alongside a brand-new fully connected head to tune specifically to satellite textures.

---

## 📊 Performance & Experimental Results

All three iterations were rigorously validated using an unseen test subset consisting of 4,050 satellite images. 

### Training Configurations
* **Optimizer:** Adam with custom learning rate scheduling (`ReduceLROnPlateau`).
* **Loss:** Cross-Entropy Loss.
* **Preprocessing:** Image upscaling to $224\times224$ pixels combined with ImageNet Z-score normalization.
* **Hardware:** CUDA-enabled GPU (8GB VRAM).

### Final Metrics Matrix

| Architecture Metric | Test Accuracy | Macro Avg $F_1$-Score | Insights & Diagnostic Profiles |
| :--- | :---: | :---: | :--- |
| **Model 1 (Baseline)** | `92.02%` | `0.92` | Exhibited optimization instability and massive early overfitting before switching to ImageNet standard normalization. Struggled with structural mix-ups between visually overlapping green categories (e.g., Herbaceous Vegetation vs. Crops). |
| **Model 2 (Improved)** | `96.72%` | `0.97` | Batch Normalization and spatial augmentation drastically smoothed the learning curve, compressing misclassification rates. Achieved a near-perfect classification rate on `Forest` tiles (only 1 misclassified out of 450). |
| **Model 3 (ResNet-50)**| **`98.02%`** | **`0.98`** | Exceptional validation stability with rapid convergence. Pre-trained features effectively resolved traditional color overlaps, pulling the lowest-performing category up to an impressive $F_1$-score of `0.96`. |

---

## 🖥️ User Interface Overview
The AI Deforestation Monitoring system is deployed locally via a dark-themed, dashboard-style web interface using the **Gradio** framework.

* **Left Control Panel:** Dropdown selection for the underlying model ("Intelligence Level") paired with a standard file drag-and-drop mechanism for incoming satellite imagery tiles.
* **Center Results Panel:** Displays real-time, dynamic inference visualization mapped against confidence percentages.
* **Right Reference Guide:** Built-in categorization layout dividing outputs cleanly across Natural Forests, Agricultural Deforestation footprints, Urbanization, and Natural Non-Forest areas.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.11+
* CUDA Toolkit configured environment (highly recommended for acceleration)

### Quickstart Guide
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Rooneyish/deforestation_detection.git](https://github.com/Rooneyish/deforestation_detection.git)
   cd deforestation_detection
