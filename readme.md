# 🛡️ SFFN: Edge-Deployable Deepfake Forensic Dashboard

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![OpenVINO 2026.1](https://img.shields.io/badge/Intel-OpenVINO-orange.svg)](https://docs.openvino.ai/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)

A professional forensic dashboard for the **Spatial-Frequency Fusion Network (SFFN)**. This project demonstrates real-time deepfake detection optimized for edge devices using dual-stream feature extraction and Intel® OpenVINO™ acceleration.

## 🚀 Key Features
- **Dual-Stream Architecture:** Combines spatial textural cues (EfficientNet-B0) with latent frequency residues (2D-DFT) to detect "checkerboard" artifacts.
- **Edge Optimized:** Quantized to **INT8** precision using Quantization-Aware Training (QAT), reducing model size to **6.62 MB**.
- **High Performance:** Achieves a throughput of **498.99 FPS** on standard performance cores.
- **Forensic Explainability:** Integrated Grad-CAM visualization to highlight specific facial regions under manipulation (eyes, mouth).

## 📊 Performance Metrics
| Metric | Value |
| :--- | :--- |
| **Peak ROC-AUC (Celeb-DF v2)** | 0.9953 |
| **Real Recall (Specificity)** | 91.31% |
| **Inference Speed** | ~499 FPS |
| **Model Size (INT8)** | 6.62 MB |

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/eashwar05/SFFN_dashboard.git](https://github.com/eashwar05/SFFN_dashboard.git)
   cd SFFN_dashboard