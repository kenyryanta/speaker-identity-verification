# 🎙️ Deep Speaker: Biometric Voice Verification System

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Architecture](https://img.shields.io/badge/Architecture-ResCNN-purple.svg?style=for-the-badge)](https://arxiv.org/pdf/1705.02304.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

## 🔭 Executive Summary
This repository implements a robust **Speaker Verification** system capable of authenticating identity based on vocal characteristics. Built upon the **Deep Speaker** architecture (Baidu Research), the system maps raw audio waveforms into a high-dimensional embedding space where speakers are separable by Euclidean or Cosine distance.

Unlike Speech Recognition (STT), which decodes *what* is being said, this system decodes *who* is saying it.

## 🏗️ System Architecture

The pipeline consists of three distinct stages designed for high-fidelity audio analysis:

### 1. Preprocessing & VAD
Raw audio is ingested and passed through a **Voice Activity Detector (VAD)**.
- **Silence Trimming:** `librosa` is used to gate noise floors below -20dB.
- **Normalization:** Audio is resampled to 16kHz to match the pre-trained receptive field.

### 2. The ResCNN Model
The backbone is a **Residual Convolutional Neural Network (ResCNN)** adapted for audio.
- **Input:** MFCC (Mel-Frequency Cepstral Coefficients) maps.
- **Hidden Layers:** Stacked Residual Blocks (ResBlocks) with clipped ReLU activation.
- **Training Objective:** **Triplet Loss**, which enforces that intra-class distances (same speaker) are smaller than inter-class distances (different speakers).
- **Output:** A compact **512-dimensional embedding vector**.



### 3. Verification Logic
Identity is verified using **Cosine Similarity**:
$$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
- **Threshold ($\theta$):** `0.50` (Calibrated for low False Acceptance Rate).

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* CUDA-enabled GPU (Recommended for inference)

### Dependencies
```bash
# System-level dependencies for audio processing
sudo apt-get install libsndfile1

# Python dependencies
pip install tensorflow librosa numpy pandas seaborn scipy
