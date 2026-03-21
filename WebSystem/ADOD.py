import os
from dotenv import load_dotenv
import base64
import os
import cv2
import torch
import json
import subprocess
from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
from simple_salesforce import Salesforce
from datetime import datetime
import requests
import threading
app = Flask(__name__)

load_dotenv()

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Action Model Path (Your local fine-tuned ViT)
ACTION_MODEL_PATH = "/media/rajendraprasath-m/New Volume/Projects/Final Year Project/WebSystem/Models/vit_final_model" 
# 2. Object Model Path (Hugging Face OWL-ViT)
OBJECT_MODEL_ID = "/media/rajendraprasath-m/New Volume/Projects/Final Year Project/WebSystem/Models/owl_vit_model"
# 3. Salesforce login URL
SF_LOGIN_URL="https://uce3-dev-ed.develop.my.salesforce.com/services/oauth2/token"

# Camera Setup
# CAMERAS = {
#     "cam1": {"id": 1, "location": "Main Entrance - North Gate"},
#     "cam2": {"id": 2, "location": "Parking Lot B (Underground)"},
#     "cam3": {"id": 3, "location": "Lobby Area"},
#     "cam4": {"id": 4, "location": "Cafeteria Hallway"},
#     "cam5": {"id": 5, "location": "Back Alley / Loading Dock"}
# }
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance", "lat": 12.9716, "lng": 77.5946}, # Example: Bangalore
    "cam2": {"id": 2, "location": "Parking Lot B", "lat": 13.0827, "lng": 80.2707}, # Example: Chennai
    "cam3": {"id": 3, "location": "Lobby Area", "lat": 19.0760, "lng": 72.8777},    # Example: Mumbai
    "cam4": {"id": 4, "location": "Cafeteria", "lat": 28.6139, "lng": 77.2090},     # Example: Delhi
    "cam5": {"id": 5, "location": "Back Alley", "lat": 22.5726, "lng": 88.3639}     # Example: Kolkata
}

# --- LOAD AI MODELS ---
device = 0 if torch.cuda.is_available() else -1
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

try:
    # A. Load ViT Action Model (Local)
    print(f"[INFO] Loading Action Model from: {ACTION_MODEL_PATH}...")
    action_processor = AutoImageProcessor.from_pretrained(ACTION_MODEL_PATH)
    action_model = AutoModelForImageClassification.from_pretrained(ACTION_MODEL_PATH).to("cuda" if device == 0 else "cpu")
    action_model.eval()
    
    id2label = action_model.config.id2label
    violence_label_name = next((n for n in id2label.values() if "viol" in n.lower() and "non" not in n.lower()), "Violence")

    # B. Load OWL-ViT Object Model (Remote/Cached)
    print(f"[INFO] Loading Object Model (OWL-ViT)...")
    object_detector = pipeline(model=OBJECT_MODEL_ID, task="zero-shot-object-detection", device=device)

    print(f"[SUCCESS] Hybrid Transformer Pipeline Ready. Violence Label: {violence_label_name}")
except Exception as e:
    print(f"[CRITICAL ERROR] Model Setup failed: {e}")

# --- SALESFORCE CONNECTOR ---
def send_to_salesforce(camera_id, location, confidence, threat_type, weapons,video_path):
    try:
        print("[SALESFORCE] Authenticating with OAuth...")

        payload = {
            "grant_type": "client_credentials",
            "client_id":os.getenv("SF_CONSUMER_KEY"),
            "client_secret":os.getenv("SF_CONSUMER_SECRET")
          }

        auth_response = requests.post(SF_LOGIN_URL, data=payload)
        

        #print("STATUS CODE:", auth_response.status_code)
        #print("RAW RESPONSE:", auth_response.text)

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
            "Weapon__c":weapons,
            'Video_ID__c': video_version_id,
            'Latitude__c': camera_info['lat'],
            'Longitude__c': camera_info['lng'],
        })

        print("[SUCCESS] Salesforce Record Created")
        return True

    except Exception as e:
        print("[SALESFORCE ERROR]", e)
        return False


# # --- HYBRID ANALYSIS ENGINE ---
#     def analyze_video_hybrid(video_path):
#         vs = cv2.VideoCapture(video_path)
#         fps = vs.get(cv2.CAP_PROP_FPS) or 30
        
#         # Analyze 1 frame every 1.5 seconds to balance speed/accuracy
#         frame_interval = int(fps * 1.5) 
#         frame_count = 0
        
#         violence_found = False
#         max_action_conf = 0.0
#         detected_weapons = set()
        
#         # Zero-Shot labels for Indian Context
#         threat_labels = ["knife","axe","machete","rod","rifle","pistol"]

#         while True:
#             ret, frame = vs.read()
#             if not ret: break
            
#             frame_count += 1
#             if frame_count % frame_interval == 0:
#                 # --- 1. ACTION DETECTION (ViT) ---
#                 vit_frame = cv2.resize(frame, (224, 224))
#                 img_rgb = cv2.cvtColor(vit_frame, cv2.COLOR_BGR2RGB)
#                 inputs = action_processor(images=Image.fromarray(img_rgb), return_tensors="pt").to("cuda" if device == 0 else "cpu")
                
#                 with torch.no_grad():
#                     outputs = action_model(**inputs)
#                     probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#                     score, pred_idx = torch.max(probs, dim=-1)
                
#                 if id2label[pred_idx.item()] == violence_label_name and score.item() > 0.65:
#                     violence_found = True
#                     max_action_conf = max(max_action_conf, score.item())

#                 # --- 2. WEAPON DETECTION (OWL-ViT) ---
#                 # Resize for speed (Critical for OWL-ViT)
#                 owl_frame = cv2.resize(frame, (480, 480))
#                 owl_rgb = cv2.cvtColor(owl_frame, cv2.COLOR_BGR2RGB)
                
#                 # Using low threshold (0.12) because Zero-Shot is very strict
#                 obj_results = object_detector(Image.fromarray(owl_rgb), candidate_labels=threat_labels, threshold=0.12)
                
#                 for res in obj_results:
#                     detected_weapons.add(res['label'].upper())

#         vs.release()
#         weapons_str = ", ".join(detected_weapons) if detected_weapons else ""
#         return violence_found, max_action_conf, weapons_str
def analyze_video_hybrid(video_path):
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS) or 30
    
    # Analyze 1 frame every 1.5 seconds
    frame_interval = int(fps * 0.5)
    frame_count = 0
    
    violence_found = False
    max_action_conf = 0.0
    detected_weapons = set()
    
    threat_labels = [
    "person holding a knife",
    "person holding a machete",
    "person holding a metal rod",
    "person holding a pistol",
    "person holding a stick",
    "sharp weapon",
    "dangerous weapon",
]

    while True:
        ret, frame = vs.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # -------------------------
        # 1️⃣ ACTION DETECTION (ViT)
        # -------------------------
        vit_frame = cv2.resize(frame, (224, 224))
        img_rgb = cv2.cvtColor(vit_frame, cv2.COLOR_BGR2RGB)

        inputs = action_processor(
            images=Image.fromarray(img_rgb),
            return_tensors="pt"
        ).to("cuda" if device == 0 else "cpu")

        with torch.no_grad():
            outputs = action_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score, pred_idx = torch.max(probs, dim=-1)

        action_label = id2label[pred_idx.item()]

        if action_label == violence_label_name and score.item() > 0.65:
            violence_found = True
            max_action_conf = max(max_action_conf, score.item())

            owl_frame = cv2.resize(frame, (640,640))
            owl_rgb = cv2.cvtColor(owl_frame, cv2.COLOR_BGR2RGB)

            obj_results = object_detector(
                Image.fromarray(owl_rgb),
                candidate_labels=threat_labels,
                threshold=0.12
            )

            for res in obj_results:
                detected_weapons.add(res['label'].upper())

    vs.release()

    weapons_str = ", ".join(detected_weapons) if detected_weapons else ""

    return violence_found, max_action_conf, weapons_str

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/process_feed', methods=['POST'])
def process_feed():
    file = request.files['video']
    camera_key = request.form['camera_id']
    filepath = os.path.join(UPLOAD_FOLDER,file.filename)
    file.save(filepath)

    print(f"[INFO] Hybrid Analysis for {camera_key}...")
    is_violent, confidence, weapons = analyze_video_hybrid(filepath)
    # 

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})
    
    if is_violent or weapons != "":
        status = "ALERT"
        msg = f"Action: {confidence:.0%} Violence" if is_violent else "Suspicious Behavior"
        if weapons: msg += f" | Weapons: {weapons}"
        # file_path_str=os.path.abspath(file)
        # print(file_path_str)
        # Cloud Alert
        thread = threading.Thread(target=send_to_salesforce, args=(
            camera_key, 
            camera_info['location'], 
            confidence, 
            msg, 
            weapons, 
            filepath
        ))
        thread.start()
        print(f"[INFO] Local Analysis Complete. Salesforce upload pushed to background thread.")
        #send_to_salesforce(camera_key, camera_info['location'], confidence, msg,weapons,filepath)
    else:
        status = "SAFE"
        msg = "Normal Activity"
    # os.remove(filepath)
    return jsonify({
        "status": status,
        "label": "Violence" if is_violent else "NonViolence",
        "weapons": weapons if weapons else "NONE",
        "message": msg,
        "location": camera_info['location'],
        "confidence": f"{confidence:.2%}" if is_violent else "N/A",
        "camera_id": camera_key
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)