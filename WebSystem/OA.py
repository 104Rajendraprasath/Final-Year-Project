import os
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
SF_ORG_ALIAS = "myGPUOrg"
SF_LOGIN_URL="https://uce3-dev-ed.develop.my.salesforce.com/services/oauth2/token"
WEAPON_CLASSES = ["gun", "knife", "pistol", "rifle", "weapon"]
# Camera Setup
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance - North Gate"},
    "cam2": {"id": 2, "location": "Parking Lot B (Underground)"},
    "cam3": {"id": 3, "location": "Lobby Area"},
    "cam4": {"id": 4, "location": "Cafeteria Hallway"},
    "cam5": {"id": 5, "location": "Back Alley / Loading Dock"}
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

    # 2. Load Object Model (YOLOv8 - better for weapons than DETR)
    print(f"[INFO] Loading Object Model (YOLOv8)...")
    object_model = YOLO(YOLO_MODEL_PATH) # Standard model detects knives; can replace with custom weights
    ##if(object_model.):
        ##print("Error happens in loading yolo model")
    print(f"[SUCCESS] All models loaded. Target Label: {violence_label_name}")
except Exception as e:
    print(f"[CRITICAL ERROR] AI Setup failed: {e}")

# --- SALESFORCE CONNECTOR ---
def send_to_salesforce(camera_id, location, confidence, threat_type, weapons):
    try:
        print("[SALESFORCE] Authenticating with OAuth...")

        payload = {
            "grant_type": "client_credentials",
            
          }

        auth_response = requests.post(SF_LOGIN_URL, data=payload)
        

        #print("STATUS CODE:", auth_response.status_code)
        #print("RAW RESPONSE:", auth_response.text)

        auth_data = auth_response.json()
        

        access_token = auth_data["access_token"]
        instance_url = auth_data["instance_url"]

        sf = Salesforce(instance_url=instance_url, session_id=access_token)

        sf.Security_Alert__c.create({
            "Camera_ID__c": camera_id,
            "Location__c": location,
            "Confidence__c": float(confidence * 100),
            "Status__c": "New",
            "Threat_Type__c": threat_type,
            "Weapon__c":weapons
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
            # --- 1. WEAPON DETECTION (YOLO) ---
            # Checks for 'knife', 'scissors', etc.
            obj_results = object_model(frame, conf=0.4, verbose=False)
            for r in obj_results:
                for box in r.boxes:
                    #cls_name = object_model.names[int(box.cls[0])]
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    label = object_model.names[class_id]

                    detected_weapons.add(label)
                    #if cls_name in ['knife','pistol']: # Base COCO weapon classes
                        #detected_weapons.add(cls_name.upper())

            # --- 2. ACTION DETECTION (ViT) ---
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
    weapons_str = ", ".join(detected_weapons) if detected_weapons else ""
    return violence_found, max_confidence, weapons_str

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/process_feed', methods=['POST'])
def process_feed():
    file = request.files['video']
    camera_key = request.form['camera_id']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"[INFO] Analyzing {camera_key}...")
    is_violent, confidence, weapons = analyze_video(filepath)
    os.remove(filepath)

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})
    
    # ALERT LOGIC: Trigger if Violence Found OR Weapon Found
    if is_violent or weapons != "":
        status = "ALERT"
        threat_type = f"Violence ({confidence:.0%})" if is_violent else ""
        if weapons: threat_type += f" | WEAPON: {weapons}"
        
        # Trigger Salesforce Cloud Alert
        send_to_salesforce(camera_key, camera_info['location'], confidence, threat_type,weapons)
    else:
        status = "SAFE"
        threat_type = "Normal Activity"

    return jsonify({
        "status": status,
        "label": "Violence" if is_violent else "NonViolence",
        "weapons": weapons if weapons else "NONE",
        "message": threat_type,
        "location": camera_info['location'],
        "confidence": f"{confidence:.2%}" if is_violent else "N/A",
        "camera_id": camera_key
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)