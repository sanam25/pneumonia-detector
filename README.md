# 🩺 MedVision AI — Pneumonia Detection System

## 📌 Overview

MedVision AI is a deep learning–based web application that detects pneumonia from chest X-ray images.
It uses a CNN model deployed via FastAPI and containerized with Docker, with a frontend hosted on Netlify.

---

## 🚀 Live Demo

* 🌐 **Frontend (Main App)**
  https://zingy-sundae-3b7b26.netlify.app/

* ⚙️ **Backend API (Docs)**
  https://pneumonia-detector-zl1u.onrender.com/docs

---

## 🧠 Model Details

* Architecture: ResNet50 (Transfer Learning)
* Input Size: 224 × 224 RGB images
* Output: Binary classification (NORMAL / PNEUMONIA)
* Framework: TensorFlow / Keras

---

## 🏗️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** FastAPI
* **ML Framework:** TensorFlow / Keras
* **Deployment:** Docker + Render
* **Hosting (Frontend):** Netlify

---

## 📂 Project Structure

```
pneumonia-detector/
│
├── backend/
│   ├── app.py
│   ├── Dockerfile
│   └── model.weights.h5
│
├── frontend/
│   ├── index.html
│   └── assets/
│       └── logo.png
│
└── README.md
```

---

## ⚙️ Installation (Local Setup)

### 1. Clone repository

```
git clone https://github.com/sanam25/pneumonia-detector.git
cd pneumonia-detector
```

---

### 2. Run backend

```
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

---

### 3. Open frontend

Open:

```
frontend/index.html
```

---

## 🐳 Docker Setup

### Build image

```
docker build -t pneumonia-app .
```

### Run container

```
docker run -p 8000:8000 pneumonia-app
```

---

## 📊 Features

* Upload chest X-ray images
* Real-time prediction
* Confidence score
* Severity classification
* Medical advice output

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should not be used for real medical diagnosis.

---

## 👨‍💻 Author

* Your Name
* GitHub: https://github.com/sanam25

---

## ⭐ Acknowledgment

This project demonstrates end-to-end deployment of an AI model using modern web technologies.
