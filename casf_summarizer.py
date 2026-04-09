import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class CASFSystem:
    def __init__(self, video_path, gt_path, output_dir):
        self.video_path = video_path
        self.gt_path = gt_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # CASF Manager: 3-Tier Regime Classification
        if "DeepSea" in video_path:
            self.regime = "DEEP_SEA"
        elif "Lake" in video_path:
            self.regime = "LAKE_WATER"
        else:
            self.regime = "OPEN_WATER"
            
        self.salience_signals = []
        self.frame_indices = []

    def casf_feature_extraction(self, frame):
        """Adaptive processing based on environment physics"""
        
        if self.regime == "LAKE_WATER":
            # Technique C (NOVELTY): Turbidity Penetration via CLAHE
            # Extracts the RED channel and stretches contrast to see through fog
            b, g, r = cv2.split(frame)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_r = clahe.apply(r)
            _, mask = cv2.threshold(enhanced_r, 150, 255, cv2.THRESH_BINARY)
            
        elif self.regime == "OPEN_WATER":
            # Technique A: Dominant Color Suppression
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            mask = cv2.absdiff(gray, blurred)
            _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
            
        else:
            # Technique B: Photometric Reflection Logic (Deep Sea)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
        # Signal Energy calculation for Temporal Graph Construction
        energy = (np.sum(mask) / (mask.shape[0] * mask.shape[1])) * 100
        return energy, mask

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        predictions = []
        
        print(f"--- Starting CASF Engine | Regime: {self.regime} ---")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            f_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            energy, mask = self.casf_feature_extraction(frame)
            
            # Decision Logic: Threshold calibrated to ignore background noise
            is_salient = 1 if energy > 5.2 else 0 
            
            self.salience_signals.append(energy)
            self.frame_indices.append(f_id)
            predictions.append({'Frame': f_id, 'Salience_Score': energy, 'Is_Salient': is_salient})

            # Visualization of the Signal Processing Pipeline
            cv2.imshow(f"CASF Input ({self.regime})", frame)
            cv2.imshow("Regime Mask (Filtered)", mask)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(f"{self.output_dir}/CASF_Prediction_Log.csv", index=False)
        self.generate_analytical_plots(pred_df)

    def generate_analytical_plots(self, pred_df):
        """Mathematical Graph Representation of the Summarization Process"""
        gt_raw = pd.read_csv(self.gt_path)
        max_f = self.frame_indices[-1]
        gt_series = np.zeros(max_f + 1)
        
        # Parse range-based Ground Truth
        for _, row in gt_raw.iterrows():
            if str(row['Object']).upper() != 'NIL':
                gt_series[int(row['Start_Frame']):int(row['End_Frame'])+1] = 1

        plt.figure(figsize=(14, 8))
        
        # Plot 1: Temporal Salience Graph
        plt.subplot(2, 1, 1)
        plt.plot(self.frame_indices, self.salience_signals, label='Signal Energy (CASF)', color='cyan')
        plt.fill_between(range(len(gt_series)), 0, gt_series * max(max(self.salience_signals), 6.0), 
                         alpha=0.2, color='gray', label='Ground Truth (Events)')
        
        # Corrected Visual Threshold Line
        plt.axhline(y=5.2, color='red', linestyle='--', label='Summarization Threshold')
        
        plt.title(f"Temporal Salience Graph - {self.regime} Regime")
        plt.ylabel("Activity Signal Energy")
        plt.legend()

        # Final Processing Scores
        merged = pred_df.copy()
        merged['GT'] = gt_series[merged['Frame']]
        
        tp = ((merged['GT'] == 1) & (merged['Is_Salient'] == 1)).sum()
        fp = ((merged['GT'] == 0) & (merged['Is_Salient'] == 1)).sum()
        fn = ((merged['GT'] == 1) & (merged['Is_Salient'] == 0)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Plot 2: Metrics Dashboard
        plt.subplot(2, 1, 2)
        plt.bar(['Precision', 'Recall', 'F1-Score'], [prec, rec, f1], color=['#2ecc71', '#e67e22', '#e74c3c'])
        plt.ylim(0, 1.1)
        plt.title("CASF Summarization Accuracy Metrics")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/CASF_Final_Analysis.png")
        plt.show()

# --- EXECUTION ---
# Change these paths to test different regimes
VIDEO = "data/1_Raw_Videos/Deep_Sea/V31_DeepSea.mp4"
GT = "data/3_Ground_Truth/Deep_Sea_GT/V31_DeepSea.csv"
OUT = "data/4_Results"

CASFSystem(VIDEO, GT, OUT).run()