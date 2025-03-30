# Channel Code Type Recognition 

This repository contains the implementation and resources for our work *"Deep Learning for Channel Code Type Recognition"*, with partial release for verification purposes:

- ✅ **Validation script, test dataset, and pre-trained weights** are currently available 
- 🔜 **Full implementation** will be released upon paper acceptance

## 🛠 Requirements

- Python 3.7+
- PyTorch 1.7.1+
- CUDA 11.0 (for GPU acceleration)

## 🏗️ Repository Structure

```
.
├── Dataset/
│   ├── data_train/       # 288,000 training samples
│   ├── data_val/         # 96,000 validation samples
│   ├── data_test/        # 96,000 test samples
│   └── data.csv          # Metadata with labels & SNR
└── CCTRNet_Code/
    ├── checkpoints/      # Pretrained models
    ├── data/             # Data loaders
    ├── models/           # Network architectures
    ├── utils/            # Helper functions
    ├── config.py         # All configuration parameters
    ├── test.py           # Evaluation script
    └── test.sh           # One-click verification
```

## 📁 Dataset

### Technical Specifications

| Parameter         | Specification                                        |
| ----------------- | ---------------------------------------------------- |
| **Code Types**    | BCH, LDPC, Polar, Convolutional, Turbo               |
| **Total Samples** | 480,000 (Train:Val:Test = 6:2:2)                     |
| **SNR Range**     | -10:2:20 dB                                          |
| **Format**        | Each sample is a `.txt` file containing encoded data |
| **Sample Size**   | 2048                                                 |

### Access Options

🔗 **Direct Download**:  
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Dataset-4285F4?logo=google-drive&style=flat)](https://drive.google.com/drive/folders/14QM4gW4BMDXNklUL3pYhRxTHswrXb9eb?usp=sharing)  

🛠 **Programmatic Access**:

```bash
# Using rclone (recommended)
rclone copy gdrive:Dataset ./Dataset -P

# Using Google Drive API
# See: https://developers.google.com/drive/api
```
