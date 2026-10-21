# Google Colab Quickstart: SSL-AASIST Training & Fine-Tuning

This guide provides the exact step-by-step instructions for running **SSL-AASIST (Phase 4 & Phase 5)** on Google Colab with GPU acceleration (T4 / V100 / A100).

---

## 1. Runtime Setup
In Google Colab, select:
**Runtime** > **Change runtime type** > **T4 GPU** (or A100 if Colab Pro).

Verify GPU availability:
```python
!nvidia-smi
```

---

## 2. Install Dependencies & Fairseq

SSL-AASIST requires a specific pinned Fairseq version:
```bash
# 1. Core deep learning & audio packages
!pip install -q torch torchvision torchaudio soundfile librosa scikit-learn tqdm

# 2. Clone and install compatible Fairseq commit
!git clone https://github.com/pytorch/fairseq
%cd fairseq
!git checkout a54021305d6b3c4c5959ac9395135f63202db8f1
!pip install --editable ./
%cd /content
```

---

## 3. Clone Repository & Upstream SSL-AASIST

```bash
# 1. Clone the multimodal deepfake detection repository
!git clone https://github.com/Abhijeet12N/Multimodal-deepfake.git /content/Multimodal-deepfake
%cd /content/Multimodal-deepfake

# 2. Clone the upstream SSL_Anti-spoofing repository
!git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git /content/SSL_Anti-spoofing
```

---

## 4. Download Pretrained XLSR-53 Frontend

```bash
# Download Fairseq XLSR-53 weights (~1.2 GB)
!wget -O /content/xlsr_53_56k.pt "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt"
```

---

## 5. Prepare ASVspoof 2019 Dataset (or Mount Google Drive)

```python
from google.colab import drive
drive.mount('/content/drive')

# Option A: If you already have ASVspoof 2019 in Drive:
!cp /content/drive/MyDrive/Datasets/LA.zip /content/
!unzip -q /content/LA.zip -d /content/ASVspoof2019_LA

# Option B: Download directly from Edinburgh DataShare:
# !curl -o /content/LA.zip "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
# !unzip -q /content/LA.zip -d /content/ASVspoof2019_LA
```

---

## 6. Execute Training with RawBoost

Run the dedicated training script with RawBoost data augmentation:

```bash
python training/train_ssl_aasist_colab.py \
    --asvspoof_dir /content/ASVspoof2019_LA \
    --xlsr_path /content/xlsr_53_56k.pt \
    --output_dir /content/drive/MyDrive/SSL_AASIST_Checkpoints \
    --epochs 15 \
    --batch_size 14 \
    --lr 0.0001 \
    --unfreeze_epoch 10 \
    --rawboost
```

---

## 7. Expected Results & Export

- **Training Output**: Best model saved to `/content/drive/MyDrive/SSL_AASIST_Checkpoints/ssl_aasist_best.pt`.
- **Target Performance**: Expected validation EER $< 1.0\%$ on ASVspoof 2019 Dev split.
- **Copying back to Local Application**: Download `ssl_aasist_best.pt` into `Multimodal-deepfake/models/` for calibrated evaluation.
