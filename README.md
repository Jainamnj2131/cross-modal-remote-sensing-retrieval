# Cross-Modal Remote Sensing Retrieval

A deep learning project for **cross-modal image retrieval** using **Sentinel-1 (SAR)** and **Sentinel-2 (Optical)** satellite imagery from the **BigEarthNet** dataset.

The objective is to learn a shared feature space where corresponding Sentinel-1 and Sentinel-2 image pairs are mapped close together, enabling retrieval across different sensing modalities.

---

## 📌 Project Objectives

- Build a cross-modal image retrieval system.
- Learn joint embeddings for Sentinel-1 and Sentinel-2 images.
- Explore and preprocess the BigEarthNet dataset.
- Train and evaluate a Two-Tower retrieval model.
- Compare retrieval performance using standard evaluation metrics.

---

## 📂 Project Structure

```
cross-modal-remote-sensing-retrieval/
│
├── configs/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```
### Folder Description

| Folder/File | Purpose |
|--------------|---------|
| `configs/` | Stores project configuration files (to be added later). |
| `data/` | Contains the Sentinel-1, Sentinel-2, and metadata datasets. This folder is ignored by Git because the datasets are too large. |
| `docs/` | Project documentation such as setup instructions, references, and project logs. |
| `models/` | Stores model architectures and related code. |
| `notebooks/` | Jupyter notebooks used for experimentation and data exploration. |
| `outputs/` | Stores generated outputs such as images, checkpoints, and results. |
| `scripts/` | Python scripts for data processing, visualization, training, and evaluation. |
| `tests/` | Contains test scripts to verify project components. |
| `.gitignore` | Specifies files and folders that Git should ignore. |
| `README.md` | Overview of the project, setup instructions, and repository documentation. |
| `requirements.txt` | Lists the Python packages required to run the project. |
---

## 🛰 Dataset

**Dataset:** BigEarthNet

- Sentinel-1 (SAR)
- Sentinel-2 (Optical)

> The dataset is **not included** in this repository because of its large size.

---

## 🛠 Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Rasterio
- Scikit-learn
- Jupyter Notebook

Future additions:

- PyTorch
- TorchVision
- TIMM
- FAISS

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd cross-modal-remote-sensing-retrieval
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📅 Current Status

**Week 1**

- ✅ Repository created
- ✅ Project structure initialized
- ✅ Virtual environment configured
- ✅ Development environment ready
- ⏳ Dataset exploration in progress

---

## 👥 Team

Developed as part of a team project on **Cross-Modal Remote Sensing Retrieval**.

---

## 📄 License

This project is intended for educational and research purposes.