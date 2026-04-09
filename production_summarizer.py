import cv2
import numpy as np
import os

class ProductionSummarizer:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 3-Tier Regime Classification
        if "DeepSea" in video_path:
            self.regime = "DEEP_SEA"
            self.threshold = 1.0 # Tuned for Deep Sea backscatter
        elif "Lake" in video_path:
            self.regime = "LAKE_WATER"
            self.threshold = 1.5 # Can be tuned later
        else:
            self.regime = "OPEN_WATER"
            self.threshold = 1.5 # Can be tuned later
            
        # Naming the output file
        base_name = os.path.basename(video_path).split('.')[0]
        self.output_file = os.path.join(output_dir, f"{base_name}_SUMMARY.mp4")

    def casf_feature_extraction(self, frame):
        """Applies the physics-informed filter based on regime"""
        if self.regime == "LAKE_WATER":
            b, g, r = cv2.split(frame)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_r = clahe.apply(r)
            _, mask = cv2.threshold(enhanced_r, 150, 255, cv2.THRESH_BINARY)
        elif self.regime == "OPEN_WATER":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            mask = cv2.absdiff(gray, blurred)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        else: # DEEP_SEA
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. NEW: The "Marine Snow" Destroyer (Low-Pass Filter)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            
            # 2. The Goldilocks Threshold (80 instead of 50 or 180)
            _, mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)
            
        energy = (np.sum(mask) / (mask.shape[0] * mask.shape[1])) * 100
        return energy, mask

    def process_and_export(self):
        cap = cv2.VideoCapture(self.video_path)
        
        # Get original video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup the Video Writer for the new Summarized Video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for Mac/mp4
        out = cv2.VideoWriter(self.output_file, fourcc, fps, (width, height))
        
        print(f"--- 🚀 Starting Production CASF Engine ---")
        print(f"Regime: {self.regime} | Threshold: {self.threshold}")
        
        saved_frames_count = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            energy, mask = self.casf_feature_extraction(frame)
            
            # If the frame has a high enough energy spike, save it to the new video!
            if energy > self.threshold:
                out.write(frame)
                saved_frames_count += 1

            # Optional: Show what is happening live
            cv2.imshow("Live Processing (Press 'q' to stop)", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Clean up and save the file
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Calculate the Compression / Reduction Ratio
        reduction_ratio = ((total_frames - saved_frames_count) / total_frames) * 100
        
        print("\n" + "="*40)
        print("📊 FINAL SUMMARIZATION REPORT")
        print("="*40)
        print(f"Original Frames:  {total_frames}")
        print(f"Summarized Frames:{saved_frames_count}")
        print(f"Data Reduction:   {reduction_ratio:.2f}% of redundant data removed!")
        print(f"File Saved To:    {self.output_file}")
        print("="*40)

# --- EXECUTION ---
VIDEO = "data/1_Raw_Videos/Deep_Sea/V31_DeepSea.mp4"
OUT_DIR = "data/4_Results/Summarized_Videos"

summarizer = ProductionSummarizer(VIDEO, OUT_DIR)
summarizer.process_and_export()