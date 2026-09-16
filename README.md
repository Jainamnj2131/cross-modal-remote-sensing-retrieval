# Cross-Modal Remote Sensing Retrieval

A deep learning project for **cross-modal satellite image retrieval** using **Sentinel-1 (SAR)** and **Sentinel-2 (multispectral optical)** imagery from the **BigEarthNet** dataset.

The project aims to learn a shared embedding space in which corresponding Sentinel-1 and Sentinel-2 images are mapped close to each other. This allows an image from one sensing modality to retrieve its corresponding image from the other modality.

---

## 📌 Project Overview

Satellite imagery can be captured using different sensing modalities.

- **Sentinel-1 (SAR)** uses radar imaging and can capture information independent of daylight and cloud conditions.
- **Sentinel-2 (Optical)** provides multispectral optical imagery containing information about land cover and surface characteristics.

The main challenge is that these two modalities have very different visual characteristics.

This project addresses that challenge using a **Two-Tower Neural Network**.

```text
                 Sentinel-1 SAR
                      │
                      ▼
                SAR ResNet-50
                      │
                      ▼
                 SAR Features
                      │
                      ▼
                SAR Projection
                      │
                      ▼
                  512-D Vector
                      │
                      │
                 Shared Space
                      │
                      │
                  512-D Vector
                      ▲
                      │
                MS Projection
                      ▲
                      │
                MS Features
                      ▲
                      │
               MS ResNet-50
                      ▲
                      │
                 Sentinel-2
```

The model is trained using **symmetric InfoNCE contrastive loss** so that matching SAR–Sentinel-2 pairs become close in the learned embedding space while non-matching pairs are pushed apart.

---

# 🎯 Project Objectives

- Build a cross-modal satellite image retrieval system.
- Pair corresponding Sentinel-1 and Sentinel-2 satellite patches.
- Preprocess both sensing modalities consistently.
- Learn modality-specific representations using separate ResNet-50 encoders.
- Project both modalities into a common **512-dimensional embedding space**.
- Train the model using symmetric InfoNCE contrastive learning.
- Generate embeddings for retrieval.
- Use FAISS for efficient nearest-neighbor search.
- Evaluate cross-modal retrieval using **Recall@1, Recall@5, and Recall@10**.
- Develop a foundation for a future retrieval application and research evaluation.

---

# 🛰 Dataset

## BigEarthNet

The project uses paired Sentinel-1 and Sentinel-2 imagery derived from the **BigEarthNet** dataset.

### Sentinel-1

Sentinel-1 data is used as the SAR modality.

Input:

```text
2 channels
├── VV
└── VH
```

The SAR preprocessing pipeline:

1. Loads VV and VH bands.
2. Clips values to the configured dB range.
3. Applies Z-score normalization using training statistics.
4. Resizes the patch to `224 × 224`.
5. Produces a tensor of shape:

```text
(2, 224, 224)
```

### Sentinel-2

Sentinel-2 data is used as the multispectral optical modality.

The current training pipeline uses:

```text
3 channels
├── B04 — Red
├── B03 — Green
└── B02 — Blue
```

The Sentinel-2 preprocessing pipeline:

1. Loads the required bands.
2. Applies the calculated dataset statistics.
3. Resizes the patch to `224 × 224`.
4. Produces a tensor of shape:

```text
(3, 224, 224)
```

### Dataset Size Used

```text
Training pairs     : 21,000
Validation pairs   : 4,500
Test pairs         : 4,500
--------------------------------
Total              : 30,000
```

The dataset itself is **not included in this repository** because of its large size.

---

# 🧠 Model Architecture

The retrieval system uses a **Two-Tower architecture**.

Each modality has an independent ResNet-50 encoder.

```text
Sentinel-1 SAR                    Sentinel-2 Optical
     │                                   │
     ▼                                   ▼
 ResNet-50                            ResNet-50
 2 channels                           3 channels
     │                                   │
     ▼                                   ▼
 2048-D features                     2048-D features
     │                                   │
     ▼                                   ▼
Projection Head                     Projection Head
2048 → 1024 → 512                  2048 → 1024 → 512
     │                                   │
     ▼                                   ▼
  512-D embedding                    512-D embedding
     │                                   │
     └──────────────┬────────────────────┘
                    ▼
             Shared Embedding Space
```

### Encoders

Both towers use:

```text
ResNet-50
```

The Sentinel-1 encoder accepts two channels:

```text
VV + VH
```

The Sentinel-2 encoder accepts three channels:

```text
B04 + B03 + B02
```

The final ResNet classification layer is replaced with an identity layer, producing **2048-dimensional features**.

### Projection Heads

Each tower has an independent projection head:

```text
2048 → 1024 → 512
```

The resulting 512-dimensional embeddings are normalized before calculating cosine similarity.

---

# 🔥 Contrastive Learning

The model is trained using a **symmetric InfoNCE loss**.

For every batch, the correct SAR–Sentinel-2 pair is treated as the positive pair.

Other samples in the same batch act as negative examples.

The loss is calculated in both directions:

```text
SAR → Sentinel-2

Sentinel-2 → SAR
```

The two losses are averaged to produce the final training loss.

### Temperature

```text
Temperature = 0.07
```

---

# 🔍 Retrieval Pipeline

After training, the learned embeddings are used for cross-modal retrieval.

The general pipeline is:

```text
Satellite Patch
      │
      ▼
Trained Two-Tower Model
      │
      ▼
512-D Embedding
      │
      ▼
FAISS Index
      │
      ▼
Nearest-Neighbor Search
      │
      ▼
Retrieved Cross-Modal Patches
```

Two retrieval directions are evaluated:

### SAR → Sentinel-2

A Sentinel-1 SAR image is used as the query and the system retrieves the most similar Sentinel-2 images.

### Sentinel-2 → SAR

A Sentinel-2 image is used as the query and the system retrieves the most similar Sentinel-1 images.

---

# 📊 Evaluation Metrics

The primary retrieval metrics are:

- **Recall@1**
- **Recall@5**
- **Recall@10**

These measure whether the correct matching image appears within the top 1, 5, or 10 retrieved results.

FAISS is also used to measure retrieval/search performance.

---

# 📈 Week 3 Baseline

Before contrastive training, a baseline retrieval experiment was performed using the initial embeddings.

### SAR → Sentinel-2

```text
R@1  = 0.2222%
R@5  = 0.4889%
R@10 = 0.8222%
```

### Sentinel-2 → SAR

```text
R@1  = 0.4889%
R@5  = 1.5111%
R@10 = 2.1333%
```

These values provide the baseline against which the trained model will be evaluated.

> Note: Same-modal retrieval results use a different relevance definition based on shared land-cover labels and therefore should not be directly compared with the cross-modal exact-pair results above.

---

# 🚀 Week 4 Training

The first full contrastive training run has been completed.

### Training Configuration

```text
Framework          : PyTorch Lightning
Model              : Two-Tower ResNet-50
Training pairs     : 21,000
Validation pairs   : 4,500
Epochs             : 20
Physical batch     : 16
Learning rate      : 0.0001
Optimizer          : Adam
Scheduler          : CosineAnnealingLR
Temperature        : 0.07
Gradient clipping  : 1.0
Precision           : 16-bit Automatic Mixed Precision
DataLoader workers : 4
GPU                 : NVIDIA GeForce RTX 3050 Laptop GPU
```

### Training Results

```text
Final training loss       : 0.02109
Final validation loss     : 0.14130
Best validation loss      : 0.14130
Best validation epoch     : 19
Training time             : 2.0474 hours
Epochs completed          : 20 / 20
```

The validation loss successfully decreased far below the initial target of 2.0.

The trained checkpoint and training logs are generated locally and are intentionally excluded from the Git repository because model checkpoints can be very large.

---

# 📂 Project Structure

```text
cross-modal-remote-sensing-retrieval/
│
├── configs/
│   └── Configuration files
│
├── data/
│   └── Dataset files
│       └── Ignored by Git
│
├── docs/
│   └── Project documentation and references
│
├── notebooks/
│   └── Jupyter notebooks for experiments and exploration
│
├── outputs/
│   ├── metadata/
│   │   └── Dataset split and metadata files
│   │
│   ├── plots/
│   │   └── Training and experiment plots
│   │
│   ├── embeddings/
│   │   └── Generated embeddings
│   │
│   └── checkpoints/
│       └── Trained model checkpoints
│           └── Ignored by Git
│
├── reports/
│   └── Experiment and training records
│
├── scripts/
│   ├── Data preparation scripts
│   ├── Dataset verification scripts
│   ├── Baseline retrieval scripts
│   └── Training scripts
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── paired_dataset.py
│   │
│   ├── models/
│   │   └── two_tower.py
│   │
│   ├── preprocessing/
│   │   ├── Sentinel-1 preprocessing
│   │   └── Sentinel-2 preprocessing
│   │
│   └── utils/
│       ├── Configuration utilities
│       └── I/O utilities
│
├── tests/
│   └── Dataset and pipeline validation tests
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

# 🧩 Important Project Components

## `src/data/`

Contains the dataset and DataLoader implementations.

The paired training pipeline loads:

```text
SAR tensor
MS tensor
Land-cover labels
Patch ID
```

The paired pipeline was validated using real Sentinel-1 and Sentinel-2 data.

---

## `src/models/`

Contains the neural network architecture.

### `two_tower.py`

Contains:

- SAR ResNet-50 encoder
- Sentinel-2 ResNet-50 encoder
- Projection heads
- Symmetric InfoNCE loss
- Two-Tower network

---

## `src/preprocessing/`

Contains preprocessing logic and dataset statistics for both modalities.

Example statistics files include:

```text
s1_stats.json
s2_stats.json
```

These statistics are used for consistent normalization during training and evaluation.

---

## `scripts/`

Contains executable project workflows including:

- Dataset preparation
- Required subset creation
- Data validation
- Paired batch testing
- Baseline retrieval
- Model training
- Future embedding generation
- Future retrieval evaluation

---

## `outputs/`

Contains generated experiment artifacts.

Large generated files such as:

- model checkpoints
- embeddings
- datasets

are intentionally excluded from Git.

---

# 🛠 Technologies

### Core

- Python
- PyTorch
- TorchVision
- PyTorch Lightning

### Data Processing

- NumPy
- Pandas
- Rasterio
- Pillow
- SciPy
- PyArrow
- PyYAML
- tqdm

### Machine Learning

- Scikit-learn
- PyTorch
- TorchVision

### Retrieval

- FAISS

### Visualization

- Matplotlib

### Application

- Flask

### Experimentation

- Jupyter Notebook

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd cross-modal-remote-sensing-retrieval
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU training, install the appropriate CUDA-enabled PyTorch build for your NVIDIA GPU and CUDA environment.

---

# 📋 Current Project Status

## Week 1 — Repository & Environment

- ✅ Repository created
- ✅ Project structure initialized
- ✅ Virtual environment configured
- ✅ Development environment prepared

## Week 2 — Dataset & Preprocessing

- ✅ BigEarthNet dataset explored
- ✅ Sentinel-1/Sentinel-2 data requirements established
- ✅ Preprocessing pipelines implemented
- ✅ Dataset statistics calculated
- ✅ Required dataset subsets prepared

## Week 3 — Paired Pipeline & Baseline Retrieval

- ✅ 30,000 paired samples prepared
- ✅ Train/validation/test splits created
- ✅ Paired Sentinel-1/Sentinel-2 Dataset implemented
- ✅ Paired DataLoader implemented
- ✅ Real-data batch validation completed
- ✅ Two-Tower model implemented
- ✅ Baseline embeddings generated
- ✅ FAISS baseline retrieval evaluated
- ✅ Baseline R@1/R@5/R@10 recorded

## Week 4 — Contrastive Training

- ✅ PyTorch Lightning training pipeline implemented
- ✅ GPU training configured
- ✅ Mixed-precision training enabled
- ✅ 21,000 training pairs used
- ✅ 4,500 validation pairs used
- ✅ 20 epochs completed
- ✅ Best model checkpoint generated
- ✅ Training metrics recorded
- ✅ Training loss plot generated
- ✅ Training results documented

Week 5 — Retrieval Evaluation & FAISS
Day 1–2: FAISS Index Construction

Built separate FAISS gallery indices for the trained test embeddings:

SAR gallery: trained_test_sar_embeddings.npy — 4,500 × 512
MS gallery: trained_test_ms_embeddings.npy — 4,500 × 512
Used IndexFlatIP for similarity search on L2-normalized embeddings.
Verified both indices after saving and reloading successfully.
Day 3–4: Cross-Modal Retrieval Evaluation

Evaluated the trained model on the 4,500-pair test set using paired patch IDs as ground truth.

Retrieval Mode	Recall@1	Recall@5	Recall@10	Avg. Retrieval Time
SAR → MS	38.78%	64.82%	75.04%	0.0123 ms/query
MS → SAR	37.04%	63.73%	74.00%	0.0139 ms/query
Day 5–7: Same-Modal Retrieval Evaluation

Evaluated same-modal retrieval using shared land-cover labels as relevance, with the query item excluded from its own gallery.

Retrieval Mode	Recall@1	Recall@5	Recall@10	mAP@10	Avg. Retrieval Time
SAR → SAR	      53.40%	85.98%	92.49%	61.24%	       0.0118 ms/query
MS → MS	      57.84%	87.04%	93.47%	64.09%	       0.0113 ms/query

Week 5 Deliverables
Implemented trained test-set embedding extraction.
Built and verified SAR and MS FAISS indices.
Implemented cross-modal retrieval evaluation.
Implemented same-modal retrieval evaluation with mAP@10.
Generated final comparison CSV and text report.
Added Week 5 evaluation scripts and results to the repository.
Updated .gitignore to exclude generated FAISS indices and other large/generated artifacts.

Status: Week 5 completed.

---

# 👥 Team

This is a team project for **Cross-Modal Remote Sensing Retrieval**.

The project involves work across:

- Dataset preparation and validation
- Remote sensing research
- Model architecture
- Contrastive learning
- Retrieval evaluation
- Application development
- Documentation and research analysis

---

# 📄 Research & Educational Purpose

This project is developed for educational and research purposes as part of an academic project.

The system is intended to investigate the feasibility of cross-modal retrieval between SAR and optical satellite imagery and to provide a foundation for further research and application development.
