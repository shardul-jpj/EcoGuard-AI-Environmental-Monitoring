import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.geo_analysis.model import create_fire_detection_model # Keeping name for now, logic updated
from PIL import Image

def generate_dummy_data(data_dir):
    """Generates dummy images for testing the pipeline."""
    categories = ['fire', 'water', 'forest', 'land']
    for category in categories:
        path = os.path.join(data_dir, category)
        os.makedirs(path, exist_ok=True)
        # Create 10 dummy images per category
        for i in range(10):
            # Generate different colors for different categories to simulate features
            if category == 'fire':
                color = (255, 100, 0) # Orange-ish
            elif category == 'water':
                color = (0, 100, 255) # Blue-ish
            elif category == 'forest':
                color = (0, 150, 0) # Green-ish
            else:
                color = (150, 150, 150) # Grey-ish
                
            img_array = np.full((224, 224, 3), color, dtype=np.uint8)
            # Add some noise
            noise = np.random.randint(-20, 20, (224, 224, 3), dtype=np.int16)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            img.save(os.path.join(path, f'dummy_{i}.jpg'))

def train_geo_model():
    data_dir = 'data/geo_dataset'
    model_save_path = 'models/geo_model.h5'
    
    # Check if data exists, if not generate dummy data
    if not os.path.exists(data_dir):
        print("Dataset not found. Generating dummy data for demonstration...")
        generate_dummy_data(os.path.join(data_dir, 'train'))
        generate_dummy_data(os.path.join(data_dir, 'val'))
        
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=8, class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=8, class_mode='categorical'
    )

    model = create_fire_detection_model()
    
    print("Starting training...")
    model.fit(train_generator, epochs=5, validation_data=val_generator) # Increased epochs slightly
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_geo_model()
