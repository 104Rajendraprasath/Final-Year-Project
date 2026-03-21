import os
import cv2
import torch
import json
import subprocess
import time
from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
from PIL import Image
from simple_salesforce import Salesforce
from datetime import datetime
import requests

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. LOCAL PATHS
BASE_DIR = "/media/rajendraprasath-m/New Volume/Projects/Final Year Project/WebSystem/Models"
ACTION_MODEL_PATH = os.path.join(BASE_DIR, "vit_final_model")
OBJECT_MODEL_PATH = os.path.join(BASE_DIR, "owl_vit_model")
SF_LOGIN_URL="https://uce3-dev-ed.develop.my.salesforce.com/services/oauth2/token"
# Camera Setup
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance - North Gate"},
    "cam2": {"id": 2, "location": "Parking Lot B (Underground)"},
    "cam3": {"id": 3, "location": "Lobby Area"},
    "cam4": {"id": 4, "location": "Cafeteria Hallway"},
    "cam5": {"id": 5, "location": "Back Alley / Loading Dock"}
}

# --- LOAD AI MODELS ---
device = 0 if torch.cuda.is_available() else -1
print(f"[INFO] Using device: {'GPU' if device == 0 else 'CPU'}")

try:
    # A. Action Model (ViT)
    action_processor = AutoImageProcessor.from_pretrained(ACTION_MODEL_PATH)
    action_model = AutoModelForImageClassification.from_pretrained(ACTION_MODEL_PATH).to("cuda" if device == 0 else "cpu")
    action_model.eval()
    
    id2label = action_model.config.id2label
    # Correctly identify Violence label while avoiding Non-Violence
    violence_label_name = next((n for n in id2label.values() if "viol" in n.lower() and "non" not in n.lower()), "Violence")

    # B. Object Model (OWL-ViT)
    # Using the local path directly in the pipeline
    object_detector = pipeline(task="zero-shot-object-detection", model=OBJECT_MODEL_PATH, device=device)

    print(f"[SUCCESS] High-Speed Hybrid Pipeline Ready.")
except Exception as e:
    print(f"[CRITICAL ERROR] Setup failed: {e}")

# --- SALESFORCE CONNECTOR ---
def send_to_salesforce(camera_id, location, confidence, threat_type, weapons):
    try:
        print("[SALESFORCE] Authenticating with OAuth...")

        payload = {
            # "grant_type": "client_credentials",
           
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


# #HYBRID ANALYSIS ENGINE (OPTIMIZED) ---Better Dtection
# def analyze_video_hybrid(video_path):
#     vs = cv2.VideoCapture(video_path)
#     fps = vs.get(cv2.CAP_PROP_FPS) or 30
#     total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # SPEED OPTIMIZATION: Check every 2 seconds (High impact on speed)
#     frame_interval = int(fps * 1.0) 
    
#     violence_found = False
#     max_action_conf = 0.0
#     detected_weapons = set()
    
#     # ACCURAY OPTIMIZATION: Contextual Prompts (Prompt Engineering)
#     # We use sentences so the model understands the action/object relationship
#     threat_labels = [
#         "a knife", "a handheld knife", "a person holding a knife",
#         "a pistol", "a handgun",
#         "a machete", "a large blade",
#         "an axe",
#         "a rifle",
#         "a metal rod", "a stick"
#     ]
    
#     # Map long prompts back to short display labels
#     label_map = {
#         "a knife": "KNIFE", "a handheld knife": "KNIFE", "a person holding a knife": "KNIFE",
#         "a pistol": "GUN", "a handgun": "GUN",
#         "a machete": "MACHETE", "a large blade": "MACHETE",
#         "an axe": "AXE",
#         "a rifle": "RIFLE",
#         "a metal rod": "ROD", "a stick": "STICK"
#     }

#     # Enable FP16 and Inference Mode for GPU speed
#     with torch.inference_mode():
#         for fno in range(0, total_frames, frame_interval):
#             vs.set(cv2.CAP_PROP_POS_FRAMES, fno)
#             ret, frame = vs.read()
#             if not ret: break

#             # ACCURACY FIX 3: Increase resolution to 600px
#             # This is the "Sweet Spot" for detecting thin objects like knives
#             frame_resized = cv2.resize(frame, (600, 600))
#             img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(img_rgb)

#             # --- 1. ACTION DETECTION (ViT) ---
#             # Action model still uses 224
#             vit_inputs = action_processor(images=pil_img.resize((224, 224)), return_tensors="pt").to(device)
#             outputs = action_model(**vit_inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             score, pred_idx = torch.max(probs, dim=-1)
            
#             if id2label[pred_idx.item()] == violence_label_name and score.item() > 0.65:
#                 violence_found = True
#                 max_action_conf = max(max_action_conf, score.item())

#             # --- 2. WEAPON DETECTION (OWL-ViT) ---
#             # ACCURACY FIX 4: Lower threshold to 0.12
#             # Zero-shot is naturally "uncertain". 0.12 catches real objects without too much noise.
#             obj_results = object_detector(pil_img, candidate_labels=threat_labels, threshold=0.12)
            
#             if obj_results:
#                 # Instead of "Winner Takes All", we allow multiple detections 
#                 # but only if they are the highest score for THAT specific object type
#                 for res in obj_results:
#                     clean_label = label_map[res['label']]
#                     # If any weapon description hits > 15%, we count it
#                     if res['score'] > 0.15:
#                         detected_weapons.add(clean_label)

#     vs.release()
#     weapons_str = ", ".join(detected_weapons) if detected_weapons else ""
#     return violence_found, max_action_conf, weapons_str

def analyze_video_hybrid(video_path):
    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- STRIDE SETTINGS ---
    vit_stride = int(fps * 1.5)   # Check behavior every 1.5s
    owl_stride_default = int(fps * 5.0) # Background weapon sweep every 5s
    
    violence_found = False
    max_action_conf = 0.0
    detected_weapons = set()
    
    label_map = {
        "a knife": "KNIFE", "a handheld knife": "KNIFE", "a person holding a knife": "KNIFE",
        "a pistol": "GUN", "a handgun": "GUN", "a machete": "MACHETE", "a large blade": "MACHETE",
        "an axe": "AXE", "a rifle": "RIFLE", "a metal rod": "ROD", "a stick": "STICK"
    }
    threat_labels = list(label_map.keys())

    # Optimize GPU usage
    with torch.inference_mode(), torch.amp.autocast('cuda'):
        for fno in range(0, total_frames, vit_stride):
            vs.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ret, frame = vs.read()
            if not ret: break

            # --- STEP 1: FAST ACTION SCAN (ViT) ---
            # Low res (224) is very fast
            vit_frame = cv2.resize(frame, (224, 224))
            img_rgb = cv2.cvtColor(vit_frame, cv2.COLOR_BGR2RGB)
            inputs = action_processor(images=Image.fromarray(img_rgb), return_tensors="pt").to(device)
            
            outputs = action_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score, pred_idx = torch.max(probs, dim=-1)
            
            curr_action_label = id2label[pred_idx.item()]
            curr_action_score = score.item()

            # --- STEP 2: SMART GATING LOGIC ---
            # We only run the heavy OWL-ViT if ViT is suspicious OR it's time for a periodic sweep
            run_weapon_check = False
            
            if curr_action_score > 0.90:
                run_weapon_check = True # Suspected violence! Check for weapons.
                violence_found = True
                max_action_conf = max(max_action_conf, curr_action_score)
            
            if fno % owl_stride_default == 0:
                run_weapon_check = True # Periodic background check

            # --- STEP 3: HEAVY WEAPON SCAN (OWL-ViT) ---
            if run_weapon_check:
                # Use 512px instead of 600px (Faster for GPU kernels, still sharp)
                owl_frame = cv2.resize(frame, (512, 512))
                img_owl_rgb = cv2.cvtColor(owl_frame, cv2.COLOR_BGR2RGB)
                
                obj_results = object_detector(
                    Image.fromarray(img_owl_rgb), 
                    candidate_labels=threat_labels, 
                    threshold=0.12 # Maintain sensitivity
                )
                
                for res in obj_results:
                    if res['score'] > 0.15:
                        detected_weapons.add(label_map[res['label']])

    vs.release()
    weapons_str = ", ".join(detected_weapons) if detected_weapons else ""
    return violence_found, max_action_conf, weapons_str


# def analyze_video_hybrid(video_path):
#     vs = cv2.VideoCapture(video_path)
#     fps = vs.get(cv2.CAP_PROP_FPS) or 30
#     total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     # --- STRIDE SETTINGS ---
#     action_stride = int(fps * 1.0)       # Check behavior every 1.0s (Fast)
#     background_sweep_stride = int(fps * 5.0) # Check for hidden weapons every 5s (Slow)
    
#     violence_found = False
#     max_action_conf = 0.0
#     detected_weapons = set()
    
#     # Accurate labels with prompt engineering
#     label_map = {
#         "a knife": "KNIFE", "a handheld knife": "KNIFE", "a person holding a knife": "KNIFE",
#         "a pistol": "GUN", "a handgun": "GUN", "a machete": "MACHETE", 
#         "a large blade": "MACHETE", "an axe": "AXE", "a rifle": "RIFLE", 
#         "a metal rod": "ROD", "a stick": "STICK"
#     }
#     threat_labels = list(label_map.keys())

#     # Use High-Speed Inference + Mixed Precision
#     with torch.inference_mode(), torch.amp.autocast('cuda' if device == 0 else 'cpu'):
#         for fno in range(0, total_frames, action_stride):
#             vs.set(cv2.CAP_PROP_POS_FRAMES, fno) # Jump directly to frame
#             ret, frame = vs.read()
#             if not ret: break

#             # --- STEP 1: FAST BEHAVIORAL SCAN (ViT) ---
#             # Action model is very fast at 224px
#             vit_frame = cv2.resize(frame, (224, 224))
#             img_vit_rgb = cv2.cvtColor(vit_frame, cv2.COLOR_BGR2RGB)
#             inputs = action_processor(images=Image.fromarray(img_vit_rgb), return_tensors="pt").to(device)
            
#             outputs = action_model(**inputs)
#             probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             score, pred_idx = torch.max(probs, dim=-1)
            
#             curr_action_label = id2label[pred_idx.item()]
#             curr_action_score = score.item()

#             # --- STEP 2: TRIGGER LOGIC (THE SPEED ENGINE) ---
#             # We only run the heavy 600px Object detector if:
#             # 1. Behavior is suspicious (>30% confidence in violence)
#             # 2. OR it is the 5-second periodic sweep
#             is_suspicious = ("viol" in curr_action_label.lower() and curr_action_score > 0.30)
#             is_sweep_time = (fno % background_sweep_stride == 0)

#             if is_suspicious or is_sweep_time:
#                 # --- STEP 3: ACCURATE WEAPON SCAN (OWL-ViT) ---
#                 # We use 600px here as requested for high accuracy on small objects
#                 owl_frame = cv2.resize(frame, (600, 600))
#                 img_owl_rgb = cv2.cvtColor(owl_frame, cv2.COLOR_BGR2RGB)
                
#                 obj_results = object_detector(
#                     Image.fromarray(img_owl_rgb), 
#                     candidate_labels=threat_labels, 
#                     threshold=0.12 # Maintain sensitivity for Zero-Shot
#                 )
                
#                 if obj_results:
#                     # Filter and take highest score for unique objects
#                     for res in obj_results:
#                         if res['score'] > 0.15:
#                             detected_weapons.add(label_map[res['label']])

#             # Final check for high-confidence violence
#             if "viol" in curr_action_label.lower() and curr_action_score > 0.65:
#                 violence_found = True
#                 max_action_conf = max(max_action_conf, curr_action_score)

#     vs.release()
#     weapons_str = ", ".join(detected_weapons) if detected_weapons else ""
#     return violence_found, max_action_conf, weapons_str

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/process_feed', methods=['POST'])
def process_feed():
    start_time = time.time()
    file = request.files['video']
    camera_key = request.form['camera_id']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"[INFO] Hybrid Analysis started for {camera_key}...")
    is_violent, confidence, weapons = analyze_video_hybrid(filepath)
    os.remove(filepath)

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})
    process_time = time.time() - start_time
    print(f"[INFO] Finished in {process_time:.2f}s")
    
    if is_violent or weapons != "":
        status = "ALERT"
        msg = f"Action: {confidence:.0%} Violence" if is_violent else "Weapon Detected"
        if weapons: msg += f" | Objects: {weapons}"
        send_to_salesforce(camera_key, camera_info['location'], confidence if is_violent else 0.99, msg,weapons)
    else:
        status = "SAFE"
        msg = "Normal Activity"

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