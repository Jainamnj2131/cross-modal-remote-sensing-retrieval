# Project Setup Guide

This guide explains how to set up the project on a new machine.

---

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Replace `<repository-url>` with the GitHub repository URL.

---

## 2. Move into the Project Directory

```bash
cd cross-modal-remote-sensing-retrieval
```

---

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 5. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## 6. Verify Installation

Check that Python is available:

```bash
python --version
```

Check installed packages:

```bash
pip list
```

---

## Notes

- Do not upload the `venv` folder.
- Do not upload the dataset (`data/`).
- Always activate the virtual environment before working on the project.