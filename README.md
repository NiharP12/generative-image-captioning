# Image Caption Generator (ResNet50 + LSTM)

## Overview

This project implements an end-to-end **Generative AI system** that generates natural language captions from images using a combination of **Computer Vision and NLP**.

The model uses a **fine-tuned ResNet50 encoder** and an **LSTM decoder** to generate captions word-by-word.

This follows the classic **Encoder–Decoder architecture (Show and Tell)**.

---

## Architecture

### 🔹 Encoder: ResNet50 (Fine-Tuning)

* Pretrained on ImageNet
* Final fully connected layer removed
* Only layer4 is unfrozen for fine-tuning
* Outputs a **2048-d feature vector projected to embedding space**

### 🔹 Decoder: LSTM

* Embedding layer converts words → vectors
* LSTM generates captions sequentially
* Uses **teacher forcing** during training
* Uses **greedy decoding** during inference

---

## ⚙️ Tech Stack

* **Language:** Python
* **Framework:** PyTorch
* **Computer Vision:** torchvision, OpenCV
* **NLP:** LSTM
* **Frontend:** Streamlit
* **Dataset:** Flickr8k

---

## 📊 Dataset

* **Flickr8k Dataset**
* ~8,000 images

### Preprocessing

* Resize images to **224×224**
* Normalize using **ImageNet mean and std**
* Clean and tokenize captions
* Convert captions to numerical indices

---

## 🔍 Key Components

### 📌 Dataset Pipeline

* Custom Dataset class for loading image-caption pairs
* Dynamic padding using custom collate function

### 📌 Vocabulary Builder

* Removes rare words (frequency < 5)
* Adds special tokens:

  * `<pad>`, `<start>`, `<end>`, `<unk>`

### 📌 Model Design

* ResNet50 encoder with **partial fine-tuning (layer4 only)**
* LSTM decoder for sequence generation

### 📌 Training Strategy

* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam

Learning rates:

* Encoder: `1e-5`
* Decoder: `1e-4`

Uses:

* Teacher forcing
* Gradient clipping

### 📌 Inference

* Greedy decoding (word-by-word generation)
* Stops at <end> token
* BLEU score evaluation supported

---

## 📈 Evaluation Metrics

* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4

---

## 🔥 Key Highlights

* Generative AI project (text generation)
* Combines **Computer Vision + NLP**
* Transfer Learning with fine-tuning
* Modular and scalable architecture
* End-to-end pipeline (training → inference)

---

## 🧠 How It Works

```text
Image → ResNet50 → Feature Vector → LSTM → Caption
```

**Example Output:**

```text
"A dog running in the park"
```

---

## 🚀 Future Improvements

* Attention mechanism
* Beam search decoding
* Transformer-based decoder
* Upgrade dataset to Flickr30k / MS COCO
* Deployment/Steamlit
