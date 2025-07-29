# 🐱🐶 Cats vs Dogs Image Classifier

This project aims to develop an image classification API that distinguishes between cats and dogs using **Deep Learning**. The model is built using **TensorFlow** and **Convolutional Neural Networks (CNNs)**.

---

## 📌 Goals

- Train an image classification model to categorize images into two classes: **cats** and **dogs**.
- Build a deep learning pipeline that can later be deployed as an API for real-time predictions.

---

## 🛠️ Tools and Libraries

- **TensorFlow**
- **Keras CNN Layers**: `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`
- **Image Preprocessing**: `ImageDataGenerator`
- **Additional Libraries**:
  - `libarchive`
  - `pydot`
  - `cartopy`

---

## ⚙️ Setup & Installation

### 1. Install Required Libraries

```bash
!apt-get -qq install -y libarchive-dev
!pip install -U libarchive pydot cartopy
```

### 2. Download and Unzip Dataset

- Ensure the dataset file is named: `cats_and_dogs_filtered.zip`.

### 3. Mount Google Drive (for Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🧹 Data Preparation

- Images are rescaled and loaded using `ImageDataGenerator`.
- The dataset includes **3000 images**, equally split between cats and dogs.
- Preprocessing includes resizing, normalization, and shuffling.

---

## 🧠 Model Architecture

A **Sequential Convolutional Neural Network (CNN)** consisting of:

- `Conv2D` + `MaxPooling2D` layers
- `Flatten` layer to convert image matrices to feature vectors
- `Dense` layers with sigmoid activation for binary classification

---

## 🏋️ Training

- The model is trained for **10 epochs**.
- Achieved **99%+ accuracy** on the training and validation sets.

---

## 🚀 Usage

1. Prepare your dataset in the correct folder structure (`train/cats`, `train/dogs`).
2. Run the training script to build and train the CNN model.
3. Load the trained model to classify new images via an API endpoint or custom script.

---

## 📊 Results

- The model demonstrates strong performance and effectively classifies cat and dog images.
- **Accuracy: 99%+**

---

## 🔮 Future Work

- Further improve the model with more advanced architectures or data augmentation.
- Deploy the trained model as an API for real-time classification.
- Expand the dataset and model to support **multi-class classification** (e.g., other animal types).

---

## 📁 Project Structure (Example)

```
├── cats_and_dogs_filtered.zip
├── train_model.ipynb
├── classify_image.py
├── saved_model/
└── README.md
```

---

## 📬 Contact

For questions or suggestions, feel free to reach out via GitHub Issues or Fork the repo and contribute!

---
