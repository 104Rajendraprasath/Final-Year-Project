import base64
import os
import threading
import cv2
import torch
import shutil
import json
import subprocess
from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from simple_salesforce import Salesforce
from datetime import datetime
from ultralytics import YOLO 
from dotenv import load_dotenv# Ensure 'pip install ultralytics' is done
import requests
load_dotenv()

app = Flask(__name__)
load_dotenv() 
# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your ViT fine-tuned ViT model
ViT_MODEL_PATH = "/media/rajendraprasath-m/New Volume/Projects/Final Year Project/WebSystem/Models/vit_final_model" 
YOLO_MODEL_PATH="/media/rajendraprasath-m/New Volume/Projects/Final Year Project/WebSystem/Models/yolo_weapon_model_fine.pt"

SF_LOGIN_URL="https://uce3-dev-ed.develop.my.salesforce.com/services/oauth2/token"

# Camera Setup
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance", "lat": 12.9716, "lng": 77.5946}, # Example: Bangalore
    "cam2": {"id": 2, "location": "Parking Lot B", "lat": 13.0827, "lng": 80.2707}, # Example: Chennai
    "cam3": {"id": 3, "location": "Lobby Area", "lat": 19.0760, "lng": 72.8777},    # Example: Mumbai
    "cam4": {"id": 4, "location": "Cafeteria", "lat": 28.6139, "lng": 77.2090},     # Example: Delhi
    "cam5": {"id": 5, "location": "Back Alley", "lat": 22.5726, "lng": 88.3639}     # Example: Kolkata
}


# --- LOAD AI MODELS ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

try:
    # 1. Load ViT Action Model
    print(f"[INFO] Loading Action Model (ViT) from: {ViT_MODEL_PATH}...")
    processor = AutoImageProcessor.from_pretrained(ViT_MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(ViT_MODEL_PATH).to(device)
    model.eval()
    id2label = model.config.id2label
    violence_label_name = next((n for n in id2label.values() if "viol" in n.lower() and "non" not in n.lower()), "Violence")
    print(f"[SUCCESS] All models loaded. Target Label: {violence_label_name}")
except Exception as e:
    print(f"[CRITICAL ERROR] AI Setup failed: {e}")

# --- SALESFORCE CONNECTOR ---
def send_to_salesforce(camera_id, location, confidence, threat_type,video_path):
    try:
        print("[SALESFORCE] Authenticating with OAuth...")

        payload = {
            "grant_type": "client_credentials",
            "client_id":os.getenv("SF_CONSUMER_KEY"),
            "client_secret":os.getenv("SF_CONSUMER_SECRET")
          }


        auth_response = requests.post(SF_LOGIN_URL, data=payload)
        
        auth_data = auth_response.json()
        

        access_token = auth_data["access_token"]
        instance_url = auth_data["instance_url"]

        sf = Salesforce(instance_url=instance_url, session_id=access_token)
        with open(video_path, "rb") as f:
            encoded_video = base64.b64encode(f.read()).decode('utf-8')

        content_version = sf.ContentVersion.create({
            'Title': f'Evidence_{camera_id}_{datetime.now().strftime("%H%M%S")}',
            'PathOnClient': 'evidence.mp4',
            'VersionData': encoded_video,
            'IsMajorVersion': True
        })

        camera_info = CAMERAS.get(camera_id, {"location": "Unknown"})
        video_version_id = content_version['id']
        sf.Security_Alert__c.create({
            "Camera_ID__c": camera_id,
            "Location__c": location,
            "Confidence__c": float(confidence * 100),
            "Status__c": "New",
            "Threat_Type__c": threat_type,
            'Video_ID__c': video_version_id,
            'Latitude__c': camera_info['lat'],
            'Longitude__c': camera_info['lng'],
        })


        print("[SUCCESS] Salesforce Record Created")
        return True

    except Exception as e:
        print("[SALESFORCE ERROR]", e)
        return False


# --- HYBRID ANALYSIS ENGINE ---
def analyze_video(video_path):
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps) # Check 1 frame every second
    
    frame_count = 0
    violence_found = False
    max_confidence = 0.0
    detected_weapons = set()

    while True:
        ret, frame = vs.read()
        if not ret: break
        
        frame_count += 1
        if frame_count % frame_interval == 0:

            frame_resized = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            inputs = processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                score, pred_idx = torch.max(probs, dim=-1)
            
            current_label = id2label[pred_idx.item()]
            if current_label == violence_label_name and score.item() > 0.70:
                violence_found = True
                max_confidence = max(max_confidence, score.item())

    vs.release()
    return violence_found, max_confidence

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('indexAD.html', cameras=CAMERAS)

@app.route('/process_feed', methods=['POST'])
def process_feed():
    file = request.files['video']
    camera_key = request.form['camera_id']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"[INFO] Analyzing {camera_key}...")
    is_violent, confidence= analyze_video(filepath)

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})
    
    # ALERT LOGIC: Trigger if Violence Found OR Weapon Found
    if is_violent:
        status = "ALERT"
        threat_type= f"Violence ({confidence:.0%})" if is_violent else ""
        
        
        # Trigger Salesforce Cloud Alert
        thread = threading.Thread(target=send_to_salesforce, args=(
            camera_key, 
            camera_info['location'], 
            confidence, 
            threat_type, 
            filepath
        ))
        thread.start()
        print(f"[INFO] Local Analysis Complete. Salesforce upload pushed to background thread.")
    else:
        status = "SAFE"
        threat_type = "Normal Activity"

    return jsonify({
        "status": status,
        "label": "Violence" if is_violent else "NonViolence",
        "message": threat_type,
        "location": camera_info['location'],
        "confidence": f"{confidence:.2%}" if is_violent else "N/A",
        "camera_id": camera_key
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)