# 🌟 IASNLP-2025: Word-Level Stress Identification from Speech

This project proposes an approach for **word-level stress identification** from speech using **prosodic features**, assuming that the corresponding **transcribed text** is available.

---

## 📁 Dataset

The dataset consists of a CSV file with the following columns:

- 📌 **Audio Path** – Path to the audio file  
- 📜 **Transcribed Text** – Manually transcribed speech  
- 🔤 **Stress Labels** – Word-level stress annotations (e.g., stressed/unstressed)

You can access the dataset here:  
👉 [**Raw audio files for training**](https://drive.google.com/file/d/1BhZ5VNkTb7v1AK7ALYvpSfivbij4v4i-/view?usp=sharing)

---

## 📂 File Structure

| File / Directory                      | Description                                                                 |
|---------------------------------------|-----------------------------------------------------------------------------|
| `setup_env.sh`                        | Shell script to set up the development and training environment             |
| `config.py`                           | Contains all configuration parameters (paths, hyperparameters, etc.)       |
| `dataset.py`                          | Defines a custom PyTorch-compatible dataset class for loading and preprocessing audio data in **NeMo-ASR-compatible** format |
| `model.py`                            | Contains the model architecture for stress classification                   |
| `train_test.py`                       | Includes PyTorch training and evaluation loop logic                         |
| `utils.py`                            | Utility functions for audio loading and **prosodic feature extraction**     |
| `stress_classification_model.ipynb`   | Jupyter Notebook entry point to train and test the stress classifier        |

---

## 🚀 Getting Started

- Clone the repository:
  - `git clone <repo_url>`
  - `cd <repo_directory>`

- Set up the environment:
  - create a new `conda` env with `python-3.10` version
  - `chmod +x setup_nv.sh`
  - `./setup_env.sh`

- Run the notebook:
  - Open and Run `stress_classification_model.ipynb` to train the model.

---
