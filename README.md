# AI-Driven Environmental Monitoring & Disaster Prediction System

This is a unified system containing three integrated modules:
1. **Forest Fire Detection Module**: Uses Computer Vision (MobileNetV2) to detect fire in images.
2. **Crop Recommendation Module**: Uses Machine Learning (Random Forest) to recommend crops based on soil and climate parameters.
3. **Air Pollution Monitoring & Health Risk Prediction Module**: Uses Machine Learning to predict pollution levels and health risks.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the models (optional, pre-trained models will be saved in `models/`):
   ```bash
   python src/fire_detection/train.py
   python src/crop_recommendation/train.py
   python src/pollution_monitoring/train.py
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```
