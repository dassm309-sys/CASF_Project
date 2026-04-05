import cv2
import pandas as pd
import numpy as np

# --- 1. SETUP ---
video_num = "V31" # Change this to test different videos
# --- 1. SETUP ---
video_num = "V31" 
env_folder = "Deep_Sea"  # Folder name has underscores
env_name = "DeepSea"     # File name does NOT have an underscore between Deep and Sea

video_path = f'data/1_Raw_Videos/{env_folder}/{video_num}_{env_name}.mp4'
csv_path = f'data/3_Ground_Truth/{env_folder}_GT/{video_num}_{env_name}.csv'
# Load GT and Setup Video
gt_data = pd.read_csv(csv_path)
cap = cv2.VideoCapture(video_path)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

print(f"--- Visualizing Pipeline for {video_num}_{env_name} ---")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    # PHASE A: Grayscale (Signal Cleaning)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # PHASE B: Edge Detection (Structure Highlight)
    # Canny helps visualize the 'outline' of objects in the water
    edges = cv2.Canny(blurred, 50, 150)
    
    # PHASE C: Motion Detection (MOG2)
    fg_mask = bg_subtractor.apply(blurred)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    
    # Draw Detections on Original
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # DISPLAY THE PIPELINE
    cv2.imshow('1. Original + Boxes', frame)
    cv2.imshow('2. Gray Signal', gray)
    cv2.imshow('3. Edge Highlights', edges)
    cv2.imshow('4. Motion Mask', thresh)

    if cv2.waitKey(30) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()