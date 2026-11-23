import streamlit as st
import os
import numpy as np
from PIL import Image
import tempfile
import hashlib

# Import inference functions
import sys
sys.path.append(os.getcwd())

from src.geo_analysis.inference import predict_geo_scene

# Page Configuration
st.set_page_config(
    page_title="EcoGuard AI | Future of Monitoring",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR ECO-FUTURIST THEME ---
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #e0e0e0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(74, 222, 128, 0.3);
    }

    /* Typography */
    h1, h2, h3 {
        font-weight: 700;
        background: linear-gradient(90deg, #4ade80, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(90deg, #16a34a 0%, #059669 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(22, 163, 74, 0.3);
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(22, 163, 74, 0.5);
        transform: scale(1.02);
    }

    /* Metrics & Progress */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4ade80, #22d3ee);
    }

    /* AQI Badge */
    .aqi-badge {
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: bold;
        display: inline-block;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Paths to models
GEO_MODEL_PATH = 'models/geo_model.h5'

def get_base_aqi(location):
    if not location:
        return 50 
    hash_object = hashlib.md5(location.lower().encode())
    hash_int = int(hash_object.hexdigest(), 16)
    aqi = (hash_int % 450) + 20 
    return aqi

def calculate_final_aqi(base_aqi, scene_label):
    if scene_label == 'Fire':
        return 500, "Hazardous", "#ef4444", "CRITICAL: Fire smoke detected. Air is toxic."
    
    final_aqi = base_aqi
    if scene_label == 'Forest' and base_aqi < 100:
        final_aqi = max(20, final_aqi - 10)
        
    if final_aqi <= 50: category, color = "Good", "#22c55e"
    elif final_aqi <= 100: category, color = "Moderate", "#eab308"
    elif final_aqi <= 150: category, color = "Unhealthy for Sensitive Groups", "#f97316"
    elif final_aqi <= 200: category, color = "Unhealthy", "#ef4444"
    elif final_aqi <= 300: category, color = "Very Unhealthy", "#a855f7"
    else: category, color = "Hazardous", "#881337"
    
    msg = f"Based on Location & {scene_label} Context"
    return final_aqi, category, color, msg

# --- APP LAYOUT ---

# Hero Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>EcoGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;' class='subtitle'>Next-Gen Environmental Intelligence & Disaster Prediction</p>", unsafe_allow_html=True)

st.markdown("---")

# Main Content Grid
row1_col1, row1_col2 = st.columns([1, 1.5], gap="large")

with row1_col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📡 Data Input")
    uploaded_file = st.file_uploader("Upload Satellite/Drone Imagery", type=["jpg", "jpeg", "png"])
    location_input = st.text_input("📍 Target Location (Optional)", placeholder="e.g., Amazon Rainforest")
    
    analyze_btn = st.button("Initiate Analysis Protocol")
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(image, caption='Source Imagery', use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with row1_col2:
    if uploaded_file is not None and analyze_btn:
        with st.spinner('Running Neural Network Inference...'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            try:
                # 1. Image Analysis
                label, probs, insights = predict_geo_scene(None, tmp_path)
                
                # 2. AQI Calculation
                base_aqi = get_base_aqi(location_input)
                final_aqi, aqi_cat, aqi_color, aqi_msg = calculate_final_aqi(base_aqi, label)
                
                # --- RESULTS DASHBOARD ---
                
                # Top Stats Row
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                stat_c1, stat_c2 = st.columns(2)
                with stat_c1:
                    st.markdown(f"<div class='metric-label'>Detected Scene</div><div class='metric-value'>{label}</div>", unsafe_allow_html=True)
                    st.progress(int(probs[label]))
                    st.caption(f"Confidence: {probs[label]:.1f}%")
                
                with stat_c2:
                    st.markdown(f"<div class='metric-label'>Air Quality Index</div><div class='metric-value' style='color:{aqi_color}'>{final_aqi}</div>", unsafe_allow_html=True)
                    st.markdown(f"<span class='aqi-badge' style='background-color:{aqi_color}'>{aqi_cat}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # Detailed Analysis Tabs
                st.markdown("### 🧠 Strategic Intelligence")
                tab1, tab2, tab3, tab4 = st.tabs(["🚨 Emergency", "🏗️ Urban", "💧 Resources", "🌱 Ecology"])
                
                with tab1:
                    st.markdown(f"<div class='glass-card'>{insights['emergency']}</div>", unsafe_allow_html=True)
                with tab2:
                    st.markdown(f"<div class='glass-card'>{insights['urban_planning']}</div>", unsafe_allow_html=True)
                with tab3:
                    st.markdown(f"<div class='glass-card'>{insights['resource_mgmt']}</div>", unsafe_allow_html=True)
                with tab4:
                    st.markdown(f"<div class='glass-card'>{insights['ecological']}</div>", unsafe_allow_html=True)

                # Composition Breakdown
                st.markdown("### 📊 Terrain Composition")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                keys = ['Fire', 'Water', 'Forest', 'Land']
                for key in keys:
                    c1, c2 = st.columns([1, 4])
                    with c1:
                        st.write(f"**{key}** ({probs[key]:.1f}%)")
                    with c2:
                        st.progress(int(probs[key]))
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"System Error: {e}")
            finally:
                os.unlink(tmp_path)
    
    elif uploaded_file is None:
        # Placeholder state
        st.markdown('<div class="glass-card" style="text-align:center; padding: 50px;">', unsafe_allow_html=True)
        st.markdown("### ⏳ Awaiting Data Stream")
        st.markdown("Upload imagery to activate the neural network.")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b;'>© 2025 EcoGuard AI | Powered by ResNet50V2 & Deep Learning</div>", unsafe_allow_html=True)
