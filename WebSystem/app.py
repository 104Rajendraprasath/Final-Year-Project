import os
import cv2
import torch
import shutil
from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from simple_salesforce import Salesforce, SalesforceLogin
from datetime import datetime

app = Flask(__name__)





# print("1. Attempting to connect to Salesforce...")
# try:
#     sf = Salesforce(username=SF_USERNAME, 
#                     password=SF_PASSWORD, 
#                     security_token=SF_TOKEN, 
#                     consumer_key=SF_CONSUMER_KEY, 
#                     consumer_secret=SF_CONSUMER_SECRET,
#                     domain='login') # Change to 'test' if using Sandbox
#     print("   [SUCCESS] Connected to Salesforce!")
# except Exception as e:
#     print(f"   [ERROR] Connection Failed: {e}")
#     exit()


# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to your LOCAL fine-tuned model folder
# This must match the folder name you copied into this directory
LOCAL_MODEL_PATH = "vit_final_model" 

# Camera Setup (Simulated)
CAMERAS = {
    "cam1": {"id": 1, "location": "Main Entrance - North Gate"},
    "cam2": {"id": 2, "location": "Parking Lot B (Underground)"},
    "cam3": {"id": 3, "location": "Lobby Area"},
    "cam4": {"id": 4, "location": "Cafeteria Hallway"},
    "cam5": {"id": 5, "location": "Back Alley / Loading Dock"}
}

print(f"[INFO] Loading local model from: {LOCAL_MODEL_PATH}...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")


def send_to_salesforce(camera_id, location, confidence):
    try:
        # 1. Fetch the active session from Salesforce CLI
        # Make sure 'myGPUOrg' matches the alias you used in Step 3 earlier
        print(f"[INFO] Fetching Salesforce CLI session for {camera_id}...")
        result = subprocess.run(
            ['sf', 'org', 'display', '--target-org', 'myGPUOrg', '--json'],
            capture_output=True, text=True
        )
        
        org_data = json.loads(result.stdout)
        
        if org_data.get('status') == 0:
            access_token = org_data['result']['accessToken']
            instance_url = org_data['result']['instanceUrl']
            
            # 2. Connect to Salesforce using the CLI Token
            sf = Salesforce(instance_url=instance_url, session_id=access_token)
            
            # 3. Create the Incident Record
            # Ensure these API names (ending in __c) match your Salesforce Object Manager
            sf.Security_Alert__c.create({
                'Camera_ID__c': camera_id,
                'Location__c': location,
                'Confidence__c': confidence * 100, # Convert 0.98 to 98.0
                'Status__c': 'New',
                'Incident_Time__c': datetime.now().isoformat()
            })
            
            print(f"[SUCCESS] Salesforce record created for {camera_id} at {location}")
            return True
        else:
            print("[ERROR] Salesforce CLI session not found. Please re-authorize.")
            return False
            
    except Exception as e:
        print(f"[SALESFORCE ERROR] Failed to send alert: {e}")
        return False
    

try:
    # 1. Load the Processor
    processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH)
    
    # 2. Load the Model
    model = AutoModelForImageClassification.from_pretrained(LOCAL_MODEL_PATH)
    model.to(device)
    model.eval() 
    
    # 3. Get Label Mapping
    id2label = model.config.id2label
    
    # --- BUG FIX START ---
    # Old Logic: Looked for "viol" (matched both Violence and NonViolence)
    # New Logic: Look for "viol" BUT EXCLUDE "non"
    violence_label_name = next(
        (name for name in id2label.values() 
         if "viol" in name.lower() and "non" not in name.lower()), 
        None
    )
    
    # Fallback if the logic fails (Hardcode if necessary based on your logs)
    if not violence_label_name:
        # Based on your logs, your label is exactly "Violence"
        violence_label_name = "Violence"
    # --- BUG FIX END ---
    
    print(f"[SUCCESS] Model loaded.")
    print(f"[INFO] Labels available: {id2label}")
    print(f"[INFO] TARGET VIOLENCE LABEL SET TO: '{violence_label_name}'")

except Exception as e:
    print(f"[CRITICAL ERROR] Could not load local model: {e}")
    model = None
    processor = None



# --- HELPER: Analyze Video ---
def analyze_video(video_path):
    if model is None:
        return "Error", 0.0

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    frame_interval = int(fps) 
    frame_count = 0
    
    # Track the best result found
    final_label = "Unknown"
    max_confidence = 0.0
    
    violence_found = False

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        
        frame_count += 1
        if frame_count % frame_interval == 0:
            # Preprocess
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            inputs = processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            score, pred_idx = torch.max(probs, dim=-1)
            
            current_label = id2label[pred_idx.item()]
            current_score = score.item()
            
            # --- IMPROVED LOGIC ---
            # 1. If we find ACTUAL violence with high confidence, lock it in.
            if current_label == violence_label_name and current_score > 0.70:
                final_label = current_label
                max_confidence = current_score
                violence_found = True
                break # Stop immediately, we found a threat
            
            # 2. If we haven't found violence yet, keep the highest score seen so far
            # (This ensures if a video is 99% NonViolence, we return "NonViolence" with 99%)
            if not violence_found and current_score > max_confidence:
                max_confidence = current_score
                final_label = current_label

    vs.release()
    return final_label, max_confidence



@app.route('/process_feed', methods=['POST'])
def process_feed():
    if 'video' not in request.files or 'camera_id' not in request.form:
        return jsonify({"error": "Missing data"}), 400
    
    file = request.files['video']
    camera_key = request.form['camera_id']
    
    # Save temporarily
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"[INFO] Analyzing feed from {camera_key}...")
    
    # Run analysis
    label, confidence = analyze_video(filepath)
    
    # Clean up
    os.remove(filepath)

    camera_info = CAMERAS.get(camera_key, {"location": "Unknown"})

    # Determine status based on the returned label
    # (We assume violence_label_name was found globally at startup)
    if label == violence_label_name:
        status_code = "ALERT"
        message_text = "Violence Detected"
    else:
        status_code = "SAFE"
        message_text = "Situation Normal"

    return jsonify({
        "status": status_code,           # ALERT or SAFE
        "label": label,                  # "Violence" or "NonViolence"
        "message": message_text,
        "location": camera_info['location'],
        "confidence": f"{confidence:.2%}", # e.g. "98.50%"
        "camera_id": camera_key
    })

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)