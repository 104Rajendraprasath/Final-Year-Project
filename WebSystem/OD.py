import base64
import os
import threading
import cv2
import requests
import torch
import json
import subprocess
import time
from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image
from simple_salesforce import Salesforce
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. LOCAL PATHS

OBJECT_MODEL_PATH = "/media/rajendraprasath-m/New Volume/Projects/Final Year Project/WebSystem/Models/owl_vit_model"
SF_LOGIN_URL="https://uce3-dev-ed.develop.my.salesforce.com/services/oauth2/token"

# Camera Setup
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance", "lat": 12.9716, "lng": 77.5946}, # Example: Bangalore
    "cam2": {"id": 2, "location": "Parking Lot B", "lat": 13.0827, "lng": 80.2707}, # Example: Chennai
    "cam3": {"id": 3, "location": "Lobby Area", "lat": 19.0760, "lng": 72.8777},    # Example: Mumbai
    "cam4": {"id": 4, "location": "Cafeteria", "lat": 28.6139, "lng": 77.2090},     # Example: Delhi
    "cam5": {"id": 5, "location": "Back Alley", "lat": 22.5726, "lng": 88.3639}     # Example: Kolkata
}

# --- LOAD OBJECT DETECTION MODEL ---
device = 0 if torch.cuda.is_available() else -1
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

try:
    print(f"[INFO] Loading local Object Model from: {OBJECT_MODEL_PATH}...")
    object_detector = pipeline(
        task="zero-shot-object-detection", 
        model=OBJECT_MODEL_PATH, 
        device=device
    )
    print(f"[SUCCESS] Object Detection Engine Ready.")
except Exception as e:
    print(f"[CRITICAL ERROR] Setup failed: {e}")

# --- SALESFORCE CONNECTOR ---
def send_to_salesforce(camera_id, location, weapons_found,video_path):
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
            # Create record in Salesforce
        sf.Security_Alert__c.create({
                'Camera_ID__c': camera_id,
                'Location__c': location,
                # Flag as high priority for object sightings
                'Status__c': 'New',
                'Threat_Type__c': f"LETHAL WEAPON DETECTED: {weapons_found}",
                'Video_ID__c': video_version_id
            })
        print(f"[SUCCESS] Salesforce Weapon Alert Dispatched for {camera_id}.")
        return True
    except Exception as e:
        print(f"[SALESFORCE ERROR] {e}")
    return False

# --- OBJECT ANALYSIS ENGINE ---
def analyze_video_objects(video_path):
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check every 1 second (Weapon sightings are often brief)
    frame_interval = int(fps * 0.5) 
    
    detected_weapons = set()
    
    # Prompt Engineering for high accuracy in Indian Context
    label_map = {
        "a person holding a knife": "KNIFE",
        "a person holding a handgun": "PISTOL",
        "a person holding a rifle": "RIFLE",
        "a machete": "MACHETE",
        "a sharp axe": "AXE",
        "a metal rod": "ROD",
        "a wooden stick": "STICK"
    }
    threat_labels = list(label_map.keys())

    with torch.inference_mode(), torch.amp.autocast('cuda' if device == 0 else 'cpu'):
        for fno in range(0, total_frames, frame_interval):
            vs.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = vs.read()
            if not ret: break

            # Resize to 640px for high accuracy on small objects
            frame_resized = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            # --- OWL-ViT INFERENCE ---
            obj_results = object_detector(
                pil_img, 
                candidate_labels=threat_labels, 
                threshold=0.15 # Sensitivity threshold
            )
            
            if obj_results:
                # Winner takes all for the frame
                best_hit = max(obj_results, key=lambda x: x['score'])
                if best_hit['score'] > 0.18:
                    detected_weapons.add(label_map[best_hit['label']])

    vs.release()
    return list(detected_weapons)

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('indexOD.html', cameras=CAMERAS)

@app.route('/process_feed', methods=['POST'])
def process_feed():
    start_time = time.time()
    file = request.files['video']
    camera_key = request.form['camera_id']
    
    # Ensure absolute path for stability
    upload_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(upload_dir): os.makedirs(upload_dir)
    
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    print(f"[INFO] Object Detection sequence started for {camera_key}...")
    
    # Run the extracted working logic
    weapons = analyze_video_objects(filepath)
    
    # Clean up


    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})
    
    if weapons:
        status = "ALERT"
        weapons_str = ", ".join(weapons)
        message = f"WEAPON DETECTED: {weapons_str}"
        # Trigger Salesforce Cloud Alert (Async/Threaded)
        # Note: We send 0.99 as confidence for object detection alerts
        thread = threading.Thread(target=send_to_salesforce, args=(
            camera_key, camera_info['location'], message,filepath
        ))
        thread.start()
    else:
        status = "SAFE"
        weapons_str = "NONE"
        message = "No Weapons Found"

    print(f"[INFO] Finished. Result: {status} | Objects: {weapons_str}")
    if os.path.exists(filepath):
        os.remove(filepath)
    return jsonify({
        "status": status,
        "weapons": weapons_str,
        "message": message,
        "location": camera_info['location'],
        "confidence": "Object Match",
        "camera_id": camera_key
    })
    start_time = time.time()
    file = request.files['video']
    camera_key = request.form['camera_id']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"[INFO] Object Scan started for {camera_key}...")
    weapons = analyze_video_objects(filepath)
    

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})
    process_time = time.time() - start_time
    
    if weapons:
        status = "ALERT"
        weapons_str = ", ".join(weapons)
        message = f"WEAPON DETECTED: {weapons_str}"
        # Trigger Cloud Alert
        #send_to_salesforce(camera_key, camera_info['location'], weapons_str,filepath)
    else:
        status = "SAFE"
        weapons_str = "NONE"
        message = "No Weapons Found"

    print(f"[INFO] Scan finished in {process_time:.2f}s. Result: {status}")
    os.remove(filepath)
    return jsonify({
        "status": status,
        "weapons": weapons_str,
        "message": message,
        "location": camera_info['location'],
        "confidence": "Object Match",
        "camera_id": camera_key
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)