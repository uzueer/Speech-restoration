# 🧠 Speech Intelligibility Restoration using Deep Complex Convolutional Recurrent Network (DCCRN)

### 🎯 Overview

This project focuses on **restoring intelligibility and quality of speech** for **hearing aid users** in noisy environments using **deep learning-based speech enhancement**.
We use the **Deep Complex Convolutional Recurrent Network (DCCRN)** architecture to perform **end-to-end speech denoising** on the **Valentini dataset** at 48 kHz.

The model learns to map **noisy speech** to **clean speech** in the complex spectrogram domain — effectively preserving both **magnitude and phase** information, which are crucial for perceptual speech quality.

---

## 🧩 Project Structure

```
dccrn_project/
├── data/
│   └── valentina/
│       ├── clean/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── noisy/
│           ├── train/
│           ├── val/
│           └── test/
│
├── models/
│   └── dccrn_model.py
│
├── results/
│   ├── denoised_audio/
│   ├── metrics/
│   └── comparisons/
│
├── scripts/
│   ├── 1_model.py         # Model initialization
│   ├── 2_dataset.py       # Dataset loader and preprocessing
│   ├── 3_train.py         # Training script
│   ├── 4_denoise.py       # Inference and denoising
│   └── 5_compare.py       # Metric evaluation (SNR, PESQ, STOI)
│
├── complexnn.py           # Custom complex-valued neural network layers
├── conv_stft.py           # STFT/ISTFT operations
└── requirements.txt
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/speech-restoration-dccrn.git
cd speech-restoration-dccrn
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Dataset Setup

Download the **Valentini Speech Dataset (48 kHz)** and structure it as follows:

```
data/valentina/clean/{train,val,test}
data/valentina/noisy/{train,val,test}
```

---

## 🧠 Model Architecture

The **DCCRN** combines:

* **Complex-valued Convolutional Layers** to extract spectral features
* **LSTM-based Recurrent Layers** for temporal modeling
* **Deconvolution Layers** for waveform reconstruction

Key modules:

* `ComplexConv2d` & `ComplexConvTranspose2d`
* `NavieComplexLSTM`
* `ComplexBatchNorm`

All implemented in **PyTorch** using custom complex-valued operations.

---

## 🚀 Training

To train the model:

```bash
python scripts/3_train.py
```

You can configure hyperparameters (epochs, batch size, learning rate) inside the script or via CLI arguments if implemented.

Model checkpoints will be saved under:

```
results/checkpoints/
```

---

## 🎧 Denoising / Inference

To denoise noisy speech samples:

```bash
python scripts/4_denoise.py --input data/valentina/noisy/test --output results/denoised_audio
```

Cleaned speech files will be stored in:

```
results/denoised_audio/
```

---

## 📊 Evaluation Metrics

After denoising, you can compare clean vs. denoised audio using:

```bash
python scripts/5_compare.py
```

The following metrics are calculated:

* **SNR (Signal-to-Noise Ratio)**
* **PESQ (Perceptual Evaluation of Speech Quality)**
* **STOI (Short-Time Objective Intelligibility)**

---

## 📈 Results

| Metric   | Noisy Input | Denoised Output (DCCRN) |
| :------- | :---------- | :---------------------- |
| SNR (dB) | ~3.2        | **14.8**                |
| PESQ     | 1.72        | **3.24**                |
| STOI     | 0.71        | **0.92**                |

> *(Values shown for illustration; your actual results may vary based on dataset and training parameters.)*

---

## 🧾 Research Paper (In Progress)

> **Title:** “Restoring Intelligibility of Speech for Hearing Aid Users in Noisy Environments using Deep Learning”
> **Institution:** SRM University, AP
> **Model Reference:** [DeepComplexCRN (Hu et al., Interspeech 2020)](https://github.com/huyanxin/DeepComplexCRN)

---

## 🛠️ Tech Stack

| Component             | Technology                                           |
| --------------------- | ---------------------------------------------------- |
| Framework             | PyTorch                                              |
| Dataset               | Valentini Speech Dataset (48 kHz)                    |
| Model Type            | Deep Complex Convolutional Recurrent Network (DCCRN) |
| Audio Processing      | STFT / ISTFT                                         |
| Evaluation            | SNR, PESQ, STOI                                      |
| Deployment (optional) | Docker / Streamlit UI                                |

---


## 🤝 Acknowledgements

Special thanks to:

* **huyanxin** for the [Deep Complex CRN architecture](https://github.com/huyanxin/DeepComplexCRN)
* **Valentini Speech Dataset** contributors
* **SRM University AP** for providing research infrastructure

---



## 📬 Contact

**Author:** Syed Uzair
📧 [[syeduzairsnu@gmail.com](mailto:syeduzairsnu@gmail.com)]
🌐 [https://www.linkedin.com/in/syeduzairn/]

