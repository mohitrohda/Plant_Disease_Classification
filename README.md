# 🌱 Plant Disease Detection System

A full-stack deep learning project that detects plant diseases from leaf images using a CNN model. The system provides real-time predictions through a FastAPI backend and a React frontend, fully containerized and deployed using Docker on AWS EC2.

---

## 🚀 Features

- 🌿 Detects 23 different plant diseases
- 📊 Trained on 35,000+ images
- 🎯 Achieves ~94% validation accuracy
- ⚡ Fast predictions using FastAPI
- 💻 Interactive UI built with React
- 🐳 Fully containerized with Docker
- ☁️ Deployed on AWS EC2

### Deploed Model link

- http://13.48.13.229:3000/

---

##  Model Details

- Model: Convolutional Neural Network (CNN)
- Input Size: 128x128x3
- Output Classes: 23
- Techniques Used:
  - Image Resizing & Rescaling
  - Data Augmentation
  - Feature Engineering
- Framework: TensorFlow / Keras

---

## 🏗️ Project Structure

plant-disease-detection/
│
├── training/
│    ├──model.ipynb
│    └── dataset
│
├── saved_models/
│
├── api/
│   ├── main.py
│   ├── class_names.py
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   └── Dockerfile
│
└── README.md
│
└── requirements.txt

---

## ⚙️ Setup Instructions

### Clone the Repository

git clone https://github.com/mohitrohda/Plant_Disease_Classification.git
cd plant-disease-detection

### Install Requirements

pip install -r requirements


### Run Backend

cd api
uvicorn main:app --reload


### Run Frontend

cd frontend
install npm
npm start

---

##  Run using Docker image

### Frontend
docker pull mohitrohda/plant_disease_frontend_only:latest
docker run -d -p 3000:80 mohitrohda/plant_disease_frontend_only:latest

### Backend 
docker pull mohitrohda/plant_disease_api:latest
docker run -p 8000:8000 mohitrohda/plant_disease_api:latest


---

## ☁️ Deployment

- Used AWS EC2 instance
- Installed Docker
- Deployed frontend & backend containers

---

## 📦 Tech Stack

- Frontend: React.js
- Backend: FastAPI
- ML: TensorFlow / Keras
- Deployment: Docker, AWS EC2

---

## 📧 Contact

Mohit Rohda- rohdamohit@gmail.com
