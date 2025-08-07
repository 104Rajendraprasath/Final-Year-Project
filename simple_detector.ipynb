import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# --- Configuration ---
# ==============================================================================
# IMPORTANT: CHANGE THIS PATH!
#
# Option 1: To use a video file, put the full path here.
# Example: VIDEO_PATH = 'D:/MyVideos/test_video.mp4'
#
# Option 2: To use your LIVE WEBCAM, change this to the number 0.
# Example: VIDEO_PATH = 0
#
VIDEO_PATH =r"D:/Projects/Final Year Project/Data/real life violence situcd ations/Real Life Violence Dataset/Violence"
# ==============================================================================

# Path to the pre-trained model file in the same folder.
MODEL_PATH = MODEL_PATH = "D:/Projects/Final Year Project/violence_detection_model.h5"

# The classes the model was trained on. This order is correct for this model.
# The model file we are using classifies 'violence' as 0 and 'non-violence' as 1.
CLASSES = ['Violence', 'NonViolence']

# --- Main Script ---

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
    exit()

# 1. Load the pre-trained model
print("[INFO] Loading model...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load model. Ensure TensorFlow is installed correctly. Error: {e}")
    exit()

# 2. Open a pointer to the video file or webcam
print(f"[INFO] Opening video source: {VIDEO_PATH}...")
vs = cv2.VideoCapture(VIDEO_PATH)

if not vs.isOpened():
    print(f"[ERROR] Could not open video source: {VIDEO_PATH}")
    exit()

# 3. Loop over the frames from the video stream
while True:
    # Read the next frame from the file/webcam
    (grabbed, frame) = vs.read()

    # If the frame was not grabbed, we have reached the end of the stream or there's an error
    if not grabbed:
        print("[INFO] No frame read from stream. Exiting.")
        break

    # --- Frame Preprocessing ---
    # Resize the frame to 224x224 pixels, the input size expected by the model
    frame_resized = cv2.resize(frame, (224, 224))
    
    # Convert the frame to a NumPy array
    frame_array = img_to_array(frame_resized)
    
    # Add a batch dimension to the array (1, 224, 224, 3)
    frame_expanded = np.expand_dims(frame_array, axis=0)

    # --- Prediction ---
    predictions = model.predict(frame_expanded, verbose=0)
    confidence = predictions[0][0]
    
    # Determine the class based on the confidence value
    if confidence < 0.5:
        predicted_class_label = 'Violence'
        percent_confidence = (1 - confidence) 
        color = (0, 0, 255) # Red for Violence
    else:
        predicted_class_label = 'NonViolence'
        percent_confidence = confidence
        color = (0, 255, 0) # Green for NonViolence
    
    # --- Output ---
    label_text = f"{predicted_class_label}: {percent_confidence:.2%}"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("Harassment Detector", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# --- Cleanup ---
print("[INFO] Cleaning up...")
vs.release()
cv2.destroyAllWindows()