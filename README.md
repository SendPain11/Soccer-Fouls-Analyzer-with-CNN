# ⚽ Soccer Fouls Analyzer with CNN

## 📌 Final Project – Computer Engineering  
**Institut Teknologi Sepuluh Nopember (ITS)**

---

### 👤 Author
**Sendy Prismana Nurferian**  
NRP: **5024211012**
Email: **sendyprisma02@gmail.com**
LinkedIn: **https://www.linkedin.com/in/sendy-prismana-nurferian-95a27b213**

---

### 👨‍🏫 Supervisors
- **Mr. Reza Fuad Rachmadi, S.T., M.T., Ph.D.**  
- **Mr. Arta Kusuma Hernanda, S.T., M.T.**

---

### 🏫 Institution
**Department of Computer Engineering**  
Faculty of Intelligent Electrical and Informatics Technology (FTEIC)  
Institut Teknologi Sepuluh Nopember (ITS)  
Surabaya, Indonesia

---

## 📖 Project Overview

**Soccer Fouls Analyzer with CNN** is a computer vision–based application developed as a **Final Project** for the Computer Engineering undergraduate program at ITS.

This system utilizes **Deep Learning with Convolutional Neural Networks (CNN)** to analyze images of soccer fouls and classify them into specific foul categories. Based on the classification result, the system provides an **automatic referee decision recommendation**, including possible card sanctions.

The application is deployed as a **web-based interface using Streamlit**, enabling users to upload an image and receive real-time analysis results.

---

## 🎯 Objectives

- To design and implement a CNN-based image classification system for soccer foul detection
- To analyze different types of fouls and handball incidents in soccer
- To provide decision support for referee judgment using AI
- To deploy the system as an interactive web application

---

## 🧠 Classification Categories

The system classifies soccer incidents into the following categories:

1. **Handball Sengaja** (Intentional Handball)  
2. **Handball Tidak Sengaja** (Unintentional Handball)  
3. **Tackle Bersih** (Clean Tackle)  
4. **Tackle Keras** (Hard Tackle)  
5. **Tackle Ringan** (Light Tackle)

---

## 🟥🟨 Card Decision Mapping

| Classification Result | Card Decision |
|----------------------|---------------|
| Handball Sengaja     | Red Card       |
| Tackle Keras         | Red Card      |
| Handball Tidak Sengaja | Yellow Card |
| Tackle Ringan        | Yellow Card   |
| Tackle Bersih        | No Card       |

---

## 🏗️ System Architecture

The system supports **two deep learning models**:

### 🔹 TensorFlow Model
- **Architecture**: ResNet101V2  
- **Framework**: TensorFlow / Keras  

### 🔹 PyTorch Model
- **Architecture**: MobileNetV4  
- **Framework**: PyTorch + timm  

Users can select which model to use directly from the Streamlit interface.

---

## 🖥️ Application Features

- Upload soccer foul images (JPG / PNG)
- Select CNN model (TensorFlow or PyTorch)
- Real-time prediction and confidence score
- Automatic referee card decision recommendation
- Visual confidence indicator
- Download analysis results in **JPG** or **PDF** format
- Deployed on **Streamlit Cloud**

---

## 🚀 Deployment

This application is deployed using **Streamlit Cloud** with the following environment:

- **Python Version**: 3.10  
- **Model Storage**: GitHub + Git LFS  
- **Hardware**: CPU-only environment  

Large model files (`.h5` and `.pth`) are managed using **Git Large File Storage (Git LFS)** to ensure reproducibility and efficient deployment.

---

## 📦 Requirements

Main dependencies used in this project:

- Streamlit
- TensorFlow
- PyTorch
- timm
- NumPy
- Pillow
- Matplotlib
- ReportLab

Refer to `requirements.txt` for the complete list.


---

## 📌 Notes

- This project is intended for **academic and research purposes**
- The system provides **decision support**, not a definitive referee ruling
- Model performance depends on dataset quality and diversity

---

## © License

This project is licensed under the **MIT License**.

---

**© 2025 – Sendy Prismana Nurferian**  
Final Project – Computer Engineering, ITS