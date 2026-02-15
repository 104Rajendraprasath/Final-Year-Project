import os
import cv2
import torch
import shutil
import subprocess  # Added
import json        # Added
from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from simple_salesforce import Salesforce
from datetime import datetime

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your LOCAL fine-tuned model folder
LOCAL_MODEL_PATH = "vit_final_model" 

# Salesforce CLI Org Alias (Make sure this matches your 'sf org authorize' alias)
SF_ORG_ALIAS = "myGPUOrg"

# Camera Setup (Simulated)
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance - North Gate"},
    "cam2": {"id": 2, "location": "Parking Lot B (Underground)"},
    "cam3": {"id": 3, "location": "Lobby Area"},
    "cam4": {"id": 4, "location": "Cafeteria Hallway"},
    "cam5": {"id": 5, "location": "Back Alley / Loading Dock"}
}

# --- LOAD AI MODEL ---
print(f"[INFO] Loading local model from: {LOCAL_MODEL_PATH}...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

try:
    processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    model.to(device)
    model.eval() 
    
    id2label = model.config.id2label
    # Find the specific label for violence
    violence_label_name = next(
        (name for name in id2label.values() if "viol" in name.lower() and "non" not in name.lower()), 
        "Violence"
    )
    
    print(f"[SUCCESS] Model loaded. Target Label: {violence_label_name}")
except Exception as e:
    print(f"[CRITICAL ERROR] Model load failed: {e}")
    model = None

# --- SALESFORCE CONNECTOR ---
def send_to_salesforce(camera_id, location, confidence):
    try:
        print(f"[SALESFORCE] Fetching CLI session for {camera_id}...")
        # Get session from Salesforce CLI
        result = subprocess.run(
            ['sf', 'org', 'display', '--target-org', SF_ORG_ALIAS, '--json'],
            capture_output=True, text=True
        )
        org_data = json.loads(result.stdout)
        
        if org_data.get('status') == 0:
            access_token = org_data['result']['accessToken']
            instance_url = org_data['result']['instanceUrl']
            
            # Connect to Salesforce
            sf = Salesforce(instance_url=instance_url, session_id=access_token)
            
            # Create Record
            sf.Security_Alert__c.create({
                'Camera_ID__c': camera_id,
                'Location__c': location,
                'Confidence__c': float(confidence * 100), 
                'Status__c': 'New',
                'Timestamp__c': datetime.now().isoformat()
            })
            print(f"[SUCCESS] Salesforce alert sent for {camera_id}")
            return True
        else:
            print("[ERROR] CLI Session not found. Run 'sf org authorize' again.")
            return False
    except Exception as e:
        print(f"[SALESFORCE ERROR] Failed: {e}")
        return False

# --- VIDEO ANALYSIS ENGINE ---
def analyze_video(video_path):
    if model is None: return "Error", 0.0
    
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    # Process 1 frame every 2 seconds for speed
    frame_interval = int(fps * 2) 
    frame_count = 0
    final_label = "Unknown"
    max_confidence = 0.0
    violence_found = False

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed: break
        
        frame_count += 1
        if frame_count % frame_interval == 0:
            # Resize for speed (ViT uses 224x224)
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            inputs = processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score, pred_idx = torch.max(probs, dim=-1)
            
            current_label = id2label[pred_idx.item()]
            current_score = score.item()
            
            # If violence detected with > 70% confidence, stop and return
            if current_label == violence_label_name and current_score > 0.70:
                final_label = current_label
                max_confidence = current_score
                violence_found = True
                break
            
            if not violence_found and current_score > max_confidence:
                max_confidence = current_score
                final_label = current_label

    vs.release()
    return final_label, max_confidence

# --- WEB ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/process_feed', methods=['POST'])
def process_feed():
    if 'video' not in request.files or 'camera_id' not in request.form:
        return jsonify({"error": "Missing data"}), 400
    
    file = request.files['video']
    camera_key = request.form['camera_id']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"[INFO] Analyzing video from {camera_key}...")
    label, confidence = analyze_video(filepath)
    os.remove(filepath)

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})

    if label == violence_label_name:
        status_code = "ALERT"
        message_text = "Violence Detected"
        # --- TRIGGER SALESFORCE ---
        send_to_salesforce(camera_key, camera_info['location'], confidence)
    else:
        status_code = "SAFE"
        message_text = "Situation Normal"

    return jsonify({
        "status": status_code,
        "label": label,
        "message": message_text,
        "location": camera_info['location'],
        "confidence": f"{confidence:.2%}",
        "camera_id": camera_key
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)