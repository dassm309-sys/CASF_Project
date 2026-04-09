# Context-Aware Adaptive Signal Fusion (CASF) V4.0
**Robust Underwater Video Summarization & Autonomous Event Detection**

**Project Team:** Midhun Dhass, Nishanth S, P Dayasuryaa  
**Guide:** Sasithradevi A (Assistant Professor, Senior Grade 2)  
**Course:** Signal Processing (BECM301L) | School of Electronics Engineering  

## 📌 Project Overview
The rapid expansion of underwater exploration using Autonomous Underwater Vehicles (AUVs) generates massive volumes of visual data, creating a critical bottleneck in storage and acoustic transmission. Conventional terrestrial summarization algorithms fail underwater due to wavelength-dependent color absorption, turbidity, and marine snow. 

This project solves this "Big Data" problem by introducing the **CASF Framework**—a state-of-the-art, physics-informed digital signal processing engine. It autonomously compresses redundant underwater video by dynamically adapting to its environment, extracting only high-value physical events (marine life, structural anomalies) to drastically reduce bandwidth requirements.

## 🔬 Engineering Challenges & Algorithmic Evolution
During development, we encountered several unique underwater physics problems. Our code evolved to solve them mathematically:

### 1. The "Marine Snow" Paradox
* **The Problem:** High-pass luminance filters intended for Deep Sea environments were constantly triggered by tiny, high-frequency suspended particles (Marine Snow), resulting in a 0% data reduction.
* **The Solution:** We implemented **Morphological Opening** (`cv2.morphologyEx`). By applying an elliptical structural matrix, the algorithm mathematically erodes sub-pixel noise while dilating solid biological targets, structurally guaranteeing that floating particles are ignored.

### 2. The Initialization Bias (Hovering Targets)
* **The Problem:** When an event (like a stationary fish) was present from frame 1, hardcoded thresholds failed. The algorithm assumed the fish was the "background noise" and deleted the footage.
* **The Solution:** We removed hardcoded thresholds and built an **Autonomous Z-Score Anomaly Detector**. The system maintains a rolling memory buffer to calculate the ambient standard deviation of the water. It triggers a "save" only when physical structures deviate by 3 standard deviations (3σ) from the local baseline, allowing it to adapt to any environment without human tuning.

### 3. The "Jarring Cut" Problem
* **The Problem:** Standard algorithms cut the video the millisecond a target stops moving, resulting in disjointed and unwatchable summary clips.
* **The Solution:** We introduced a **Temporal Context Buffer** using a Double-Ended Queue (`deque`). The engine maintains a continuous 1-second visual memory. When an event triggers, it saves the pre-roll and post-roll footage, creating a smooth, "documentary-style" cut.

## ⚙️ The CASF Processing Pipeline
The V4.0 algorithm processes video through a 5-stage autonomous pipeline:
1. **Regime Classification:** The system reads the context and switches modes (Deep Sea, Open Water, or Lake Water).
2. **Adaptive Physics Filtering:** Applies regime-specific rules (e.g., CLAHE for turbidity penetration, Dominant Color Suppression for open oceans).
3. **Structural Noise Cleaning:** Applies Gaussian blurring and Morphological matrices to destroy water-column noise.
4. **Z-Score Triggering:** Calculates real-time signal energy and triggers recording based on statistical anomaly detection.
5. **Video Export:** Compresses the selected frames into a highly reduced `.mp4` payload.

## 📂 Directory Structure
To ensure seamless data-to-label mapping and separation of development phases:

```text
├── casf_production_engine.py  # [V4.0] Main autonomous summarization engine
├── casf_summarizer.py         # Validation engine (Generates Accuracy Graphs & F1-Scores)
├── production_summarizer.py   # Legacy V2.0 (Hardcoded thresholds)
├── fish_detector.py           # Legacy V1.0 (Real-time MOG2 visualization)
├── scripts/
│   ├── standardize_gt.py      # Cleans Excel data to CSV
│   └── fix_headers.py         # Formats column headers
└── data/
    ├── 1_Raw_Videos/          # .mp4 source files categorized by regime
    ├── 3_Ground_Truth/        # Human-annotated .csv files for algorithm validation
    └── 4_Results/             # Generated graphs, CSV logs, and final compressed .mp4s

## 📦 Dataset & Large Files
The raw underwater video dataset is hosted externally due to file size limitations.

- Total Videos: 69 clips (Deep Sea, Lake Water, Open Ocean)
- Total Size: 348 MB
- Access Link: Google Drive Dataset

> Download the `1_Raw_Videos/` and `3_Ground_Truth/` folders from the Drive link and place them inside the `data/` directory.

## 🚀 Setup & Execution Instructions
### Installation
1. Clone this repository.
2. Download the dataset folders and place them in `data/`.
3. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
pip install opencv-python pandas numpy matplotlib
```

### 1. Running the Production Engine (Video Export)
To generate a highly compressed summary video:

```bash
python casf_production_engine.py
```

This outputs a fully edited `.mp4` file and a terminal report detailing the data reduction percentage in `data/4_Results/Summarized_Videos/`.

### 2. Running the Validation Engine (Graph Generation)
To validate algorithmic accuracy against human ground truth and generate deliverables:

```bash
python casf_summarizer.py
```

This outputs temporal salience graphs and an F1-score bar chart to `data/4_Results/`.

    