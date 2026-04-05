# Underwater Video Summarization & Fish Detection
**Developer:** Midhun Dhass  
**Project:** Computer-Aided Signal Processing (CASF)

## 📌 Project Overview
This project automates the detection of marine life in underwater videos using Digital Signal Processing (DSP) and Computer Vision. It processes raw footage from three environments (Deep Sea, Fresh Water, and Surface Ocean) and validates detection accuracy against human-annotated Ground Truth data.

## 📂 Directory Structure
The project follows a parallel directory architecture to ensure seamless data-to-label mapping.

- `data/1_Raw_Videos/`: Contains .mp4 source files categorized by environment.
- `data/3_Ground_Truth/`: Contains standardized .csv files with frame-level annotations.
- `scripts/`:
    - `standardize_gt.py`: Converts raw Excel sheets to clean CSVs.
    - `fix_headers.py`: Corrects formatting issues from exported data.
    - `fish_detector.py`: The core computer vision pipeline.

## ⚙️ The Detection Pipeline
The algorithm processes video through four distinct mathematical phases:
1. **Grayscale Conversion:** Reduces noise and removes color-based lighting artifacts.
2. **Gaussian Filtering:** A low-pass filter used to eliminate "Marine Snow" noise.
3. **Canny Edge Detection:** Highlights the structural outlines of moving targets.
4. **MOG2 Background Subtraction:** Mathematically isolates moving objects from static backgrounds.

## 🎥 Dataset & Large Files
Due to GitHub's file size limitations (100MB per file), the raw underwater video dataset is hosted externally. 

- **Total Videos:** 69 Clips (Deep Sea, Fresh Water, Surface Ocean)
- **Total Size:** [348MB]
- **Access Link:** [👉 Click here to access the Google Drive Dataset](https://drive.google.com/drive/folders/10TVp3Bu8jlNJOrUMb-IpNNcy6bnolty6?usp=sharing)

### **Setup Instructions**
To run the detection algorithm locally:
1. Download the `1_Raw_Videos` folder from the link above.
2. Place it inside the `data/` directory of this project.
3. Ensure the folder structure matches the `Directory Structure` section below.

## 🚀 How to Run
1. Install dependencies: `pip install opencv-python pandas numpy`
2. Run the detector: `python fish_detector.py`
