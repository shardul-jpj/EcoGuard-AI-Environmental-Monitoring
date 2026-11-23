import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained model globally
# ResNet50V2 is more accurate than MobileNetV2
model = ResNet50V2(weights='imagenet', include_top=True)

def predict_geo_scene(model_path_ignored, img_path):
    """
    Uses pre-trained ResNet50V2 to classify the scene.
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    # Get top 20 predictions to catch subtle features
    decoded = decode_predictions(preds, top=20)[0] 
    
    # Initialize scores
    scores = {
        'Fire': 0.0,
        'Water': 0.0,
        'Forest': 0.0,
        'Land': 0.0
    }
    
    # Expanded Keyword Mappings for better coverage
    keywords = {
        'Fire': [
            'fire', 'flame', 'volcano', 'lighter', 'smoke', 'eruption', 'magma', 'lava', 'matchstick'
        ],
        'Water': [
            'lake', 'ocean', 'river', 'sea', 'dam', 'fountain', 'coast', 'beach', 'island', 'pier', 
            'boat', 'canoe', 'ship', 'liner', 'dock', 'wharf', 'breakwater', 'sandbar', 'coral', 
            'scuba', 'seashore', 'lakeside', 'geyser', 'bubble', 'cliff', 'promontory'
        ],
        'Forest': [
            'forest', 'jungle', 'tree', 'park', 'garden', 'grass', 'valley', 'mountain', 'alp', 
            'rainforest', 'wood', 'orchard', 'grove', 'wilderness', 'greenhouse', 'hedge', 'snail', 
            'mushroom', 'daisy', 'flower', 'plant', 'vegetation'
        ],
        'Land': [
            'building', 'street', 'city', 'house', 'traffic', 'bridge', 'road', 'urban', 'shop', 
            'barn', 'farm', 'corn', 'wheat', 'tractor', 'harvester', 'palace', 'castle', 'monastery', 
            'church', 'mosque', 'stupa', 'library', 'bank', 'museum', 'theater', 'cinema', 'gym', 
            'restaurant', 'bakery', 'market', 'bazaar', 'parking', 'garage', 'car', 'truck', 'bus', 
            'train', 'railway', 'station', 'airport', 'plane', 'crane', 'construction', 'rubble', 
            'ruin', 'patio', 'terrace', 'tile', 'roof', 'brick', 'wall', 'fence', 'gate', 'door', 
            'window', 'solar', 'panel', 'antenna', 'dish', 'pole', 'wire', 'asphalt', 'concrete'
        ]
    }
    
    # Map ImageNet predictions to our categories
    matched_any = False
    for i, (id, label, prob) in enumerate(decoded):
        label = label.lower()
        
        for category, tags in keywords.items():
            if any(tag in label for tag in tags):
                scores[category] += float(prob)
                matched_any = True
                
    # Fallback: If absolutely nothing matched in top 20, check the very top prediction
    if not matched_any:
        top_label = decoded[0][1].lower()
        # Heuristics for unknown
        if 'fish' in top_label or 'whale' in top_label or 'shark' in top_label:
            scores['Water'] += 0.5
        elif 'bird' in top_label or 'monkey' in top_label or 'bear' in top_label:
            scores['Forest'] += 0.5
        else:
            scores['Land'] += 0.5 # Default to Land/Urban for man-made objects
            
    # Normalize scores to percentages
    total = sum(scores.values())
    if total == 0:
        scores['Land'] = 100.0
    else:
        for k in scores:
            scores[k] = (scores[k] / total) * 100
            
    # Get top label
    label = max(scores, key=scores.get)
    
    insights = generate_detailed_insights(label, scores)
    
    return label, scores, insights

def generate_detailed_insights(label, probs):
    """
    Generates a comprehensive report based on the detected scene.
    """
    report = {
        "emergency": "",
        "urban_planning": "",
        "resource_mgmt": "",
        "ecological": ""
    }
    
    if label == 'Fire':
        report["emergency"] = "🔴 **CRITICAL**: Active fire detected. Immediate deployment of fire suppression drones and evacuation of nearby settlements is required."
        report["urban_planning"] = "⛔ **UNSAFE**: Area is currently a disaster zone. Post-fire structural integrity assessment will be needed before any activity."
        report["resource_mgmt"] = "⚠️ **Damage Control**: Prioritize protecting water sources from ash contamination."
        report["ecological"] = "🔥 **Severe Impact**: High loss of biodiversity expected. Reforestation planning should begin post-disaster."
        
    elif label == 'Water':
        report["emergency"] = "🌊 **Flood Watch**: Monitor water levels. If heavy rain is forecast, issue flood warnings to downstream areas."
        report["urban_planning"] = "⛔ **RESTRICTED**: High flood risk. Not suitable for residential or commercial construction. Ideal for parks or wetlands."
        report["resource_mgmt"] = "💧 **High Potential**: Excellent for reservoir creation, rainwater harvesting, or hydroelectric power generation."
        report["ecological"] = "🐟 **Vital Habitat**: Critical for aquatic life. Pollution control measures must be strictly enforced."
        
    elif label == 'Forest':
        report["emergency"] = "🟢 **Stable**: No immediate threat, but maintain firebreaks and monitor for illegal logging activities."
        report["urban_planning"] = "⛔ **PROTECTED**: Construction is restricted to preserve green cover. Eco-tourism development is a viable alternative."
        report["resource_mgmt"] = "🌲 **Carbon Sink**: Area is crucial for carbon sequestration. Sustainable timber harvesting possible if regulated."
        report["ecological"] = "🦜 **High Biodiversity**: Primary habitat for wildlife. Maintain connectivity with other forest patches."
        
    elif label == 'Land':
        report["emergency"] = "🟡 **Monitor**: Check for soil erosion or heat island effects during summer."
        report["urban_planning"] = "✅ **OPTIMAL**: High suitability for urbanization, metro expansion, or industrial development. Ensure proper drainage."
        report["resource_mgmt"] = "🚜 **Agriculture/Solar**: If not urbanized, suitable for farming or large-scale solar farms."
        report["ecological"] = "🏙️ **Low Impact**: Already cleared. Focus on adding green spaces (parks, vertical gardens) within development plans."
        
    return report
