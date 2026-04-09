import cv2
import numpy as np
import os
from collections import deque

class CASF_ProductionEngine:
    def __init__(self, video_path, output_dir):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Regime Classification
        if "DeepSea" in video_path:
            self.regime = "DEEP_SEA"
        elif "Lake" in video_path:
            self.regime = "LAKE_WATER"
        else:
            self.regime = "OPEN_WATER"
            
        base_name = os.path.basename(video_path).split('.')[0]
        self.output_file = os.path.join(output_dir, f"{base_name}_SMART_SUMMARY.mp4")
        
        # 2. Structural Elements for Noise Cleaning
        # A 5x5 structural matrix used to mathematically destroy marine snow
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def apply_casf_physics(self, frame):
        """Applies the regime-specific physical filters and structural cleaning."""
        if self.regime == "LAKE_WATER":
            # CLAHE on Red Channel to pierce turbidity
            _, _, r = cv2.split(frame)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(r)
            _, mask = cv2.threshold(enhanced, 150, 255, cv2.THRESH_BINARY)
            
        elif self.regime == "OPEN_WATER":
            # Dominant Background Suppression
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            mask = cv2.absdiff(gray, blurred)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
            
        else: # DEEP_SEA
            # High-Pass Filter with intense spatial blurring
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            _, mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
            
        # RESEARCH NOVELTY: Morphological Opening
        # Physically erodes scattered noise (snow) and dilates solid objects (fish)
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        
        # Calculate final energy of the cleaned signal
        energy = (np.sum(clean_mask) / (clean_mask.shape[0] * clean_mask.shape[1])) * 100
        return energy, clean_mask

    def run_engine(self):
        cap = cv2.VideoCapture(self.video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        print("\n" + "="*50)
        print(f"🌊 CASF V4.0 AUTONOMOUS SUMMARIZATION ENGINE")
        print(f"📍 Active Regime: {self.regime}")
        print("="*50)
        
        # --- The Autonomous Brain Variables ---
        energy_history = deque(maxlen=fps * 3)  # 3-second memory of background noise
        frame_buffer = deque(maxlen=fps * 1)    # 1-second visual buffer (Pre-Roll)
        
        cooldown_counter = 0     # Keeps recording (Post-Roll) after the fish leaves
        saved_frames_count = 0
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_id += 1
            
            energy, mask = self.apply_casf_physics(frame)
            energy_history.append(energy)
            frame_buffer.append(frame) # Always keep the last 1 second of footage in memory
            
            # Allow the math to stabilize for 1 second before making decisions
            if frame_id > fps:
                # Calculate the Z-Score dynamic baseline
                baseline = np.mean(energy_history)
                std_dev = np.std(energy_history)
                
                # Trigger = Average Noise + (3 * Standard Deviations)
                # We add + 0.5 to prevent math errors if the water is absolutely pitch black
                adaptive_threshold = baseline + (3 * std_dev) + 0.5
            else:
                adaptive_threshold = 999.0 # Waking up, do not trigger
                
            # EVENT TRIGGER LOGIC
            is_active_event = False
            if energy > adaptive_threshold:
                is_active_event = True
                cooldown_counter = fps * 1 # Reset the 1-second Post-Roll timer
                
            # WRITING LOGIC (The "Documentary" Cut)
            if is_active_event or cooldown_counter > 0:
                # If the buffer has older frames, flush them to the video first (Pre-Roll)
                while frame_buffer:
                    out.write(frame_buffer.popleft())
                    saved_frames_count += 1
                
                # Decrement the Post-Roll timer if the signal drops
                if not is_active_event:
                    cooldown_counter -= 1

            # Console live-feed of the adaptive math
            if frame_id % 15 == 0:
                status = "🔴 REC" if (is_active_event or cooldown_counter > 0) else "⏸  STANDBY"
                print(f"Frame {frame_id:04d}/{total_frames} | Energy: {energy:05.2f} | Trigger @ {adaptive_threshold:05.2f} | {status}")

            cv2.imshow("CASF Live Processing Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        reduction = ((total_frames - saved_frames_count) / total_frames) * 100
        
        print("\n" + "="*50)
        print("✅ SUMMARIZATION COMPLETE")
        print("="*50)
        print(f"Total Video Frames: {total_frames}")
        print(f"Summarized Frames:  {saved_frames_count}")
        print(f"Bandwidth Saved:    {reduction:.2f}% Data Reduction")
        print(f"Output File:        {self.output_file}")
        print("="*50 + "\n")

# --- EXECUTION ---
# Change this variable to test different videos
VIDEO_PATH = "data/1_Raw_Videos/Deep_Sea/V31_DeepSea.mp4"
OUTPUT_FOLDER = "data/4_Results/Summarized_Videos"

engine = CASF_ProductionEngine(VIDEO_PATH, OUTPUT_FOLDER)
engine.run_engine()