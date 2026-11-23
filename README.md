# 🌍 EcoGuard AI – Environmental Monitoring & Disaster Intelligence

**An AI-driven environmental monitoring and disaster intelligence platform built using Python, Machine Learning, Deep Learning, Computer Vision, and Streamlit.**

EcoGuard AI aims to bring environmental analysis and disaster-related intelligence into a unified interactive dashboard.

## 🚀 Project Overview

Environmental challenges such as forest fires, air pollution, and changing agricultural conditions require better monitoring and intelligent analysis.

EcoGuard AI is designed to support environmental monitoring through AI-based image analysis, machine learning models, and an interactive dashboard.

## ✨ Key Features

### 🔥 Forest Fire Detection
- Image-based fire detection using Computer Vision.
- MobileNetV2-based approach, as described in the original project documentation.

### 🌱 Crop Recommendation
- Machine Learning-based crop recommendation.
- Random Forest model approach using soil and climate parameters.

### 🌫️ Air Pollution Monitoring
- Pollution monitoring and health-risk prediction module.
- Machine Learning-based approach.

### 🛰️ Geo-Scene Analysis
- Upload satellite or drone images.
- Analyze environmental scenes.
- Display Fire, Water, Forest, and Land composition.
- Generate contextual environmental insights.

### 📊 Interactive Dashboard
- Built with Streamlit.
- Environmental analysis results.
- AQI category visualization.
- Strategic intelligence sections for emergency, urban planning, resource management, and ecology.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Dashboard | Streamlit |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV, Pillow |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Models | MobileNetV2, Random Forest |

## 📁 Project Structure

```text
EcoGuard-AI-Environmental-Monitoring/
│
├── data/
│
├── models/
│   ├── crop_model.pkl
│   ├── fire_model.h5
│   ├── geo_model.h5
│   └── pollution_model.pkl
│
├── src/
│   ├── geo_analysis/
│   ├── fire_detection/
│   ├── crop_recommendation/
│   └── pollution_monitoring/
│
├── app.py
├── requirements.txt
├── README.md
└── run_project.bat
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shardul-jpj/EcoGuard-AI-Environmental-Monitoring.git
```

### 2. Navigate to the Project Directory

```bash
cd EcoGuard-AI-Environmental-Monitoring
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🧠 Machine Learning Models

The project includes the following model artifacts:

- `crop_model.pkl`
- `fire_model.h5`
- `geo_model.h5`
- `pollution_model.pkl`

The original project documentation describes Random Forest for crop recommendation and MobileNetV2 for fire detection.

## 🔮 Future Scope

- Real-time environmental data integration.
- Live weather and air-quality APIs.
- Improved disaster prediction capabilities.
- Satellite imagery integration.
- Geographical visualization and mapping.
- Advanced AI-based environmental risk assessment.

## 🎯 Project Objective

To explore how Artificial Intelligence and Machine Learning can support environmental monitoring, disaster awareness, and intelligent decision-making through a unified platform.

## 👨‍💻 Developer

**Shardul Walunj**

GitHub: https://github.com/shardul-jpj

---

*EcoGuard AI – Building intelligent solutions for a sustainable future.*
