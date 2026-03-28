# Breast Oncology Surveillance System
Student Project for Longitudinal Analysis.

🎗️ Breast Oncology Surveillance System (BOSS)
A modular Deep Learning framework for the automated segmentation of malignancies in Ultrasound and Mammography imaging. Developed as part of the Projet Fédérateur (Software Engineering, Year 1).

📌 Project Overview
The BOSS platform addresses the challenge of "Domain Variance" in medical imaging by utilizing a Modular Expert Architecture. Instead of a single "one-size-fits-all" model, the system employs two specialized U-Net models trained specifically on the unique physics of sound waves (Ultrasound) and X-rays (Mammography).

Key Features:
Architecture: U-Net with a ResNet-34 backbone (Pre-trained on ImageNet).

Multi-Modality Support: Dedicated pipelines for Dataset 1 (US) and Dataset 2 (MG).

Adaptive Preprocessing: Automated CLAHE enhancement for Ultrasound noise reduction.

Advanced Training: Implements ReduceLROnPlateau scheduling and Hybrid Dice Loss for boundary precision.

📂 Project Structure
Plaintext
├── 📂 data/               # Raw and Processed (512x512) medical datasets
├── 📂 models/             # Trained .pth weights and evaluation logs
├── 📂 src/                # Core logic (Training, Evaluation, Data Unification)
└── app.py                 # Streamlit Clinical Dashboard
🚀 Getting Started
1. Environment Setup
Bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
2. Data Preparation
Organize your raw data in data/raw/ and run the unification script to create standardized training/testing sets:

Bash
python src/unify_datasets.py
3. Training the Experts
Train each modality separately to ensure specialized feature extraction:

Bash
python src/train.py --mode us  # Train Ultrasound Expert
python src/train.py --mode mg  # Train Mammography Expert
4. Running the Clinical UI
Launch the interactive dashboard for real-time inference:
streamlit run app.py
📊 Evaluation & Metrics
The system is evaluated based on the Dice Similarity Coefficient (DSC), ensuring that the AI-predicted masks align accurately with the ground-truth annotations provided by radiologists.

Modality,Target Metric,Preprocessing
Ultrasound,Nodules/Cysts,CLAHE + Hybrid Loss
Mammography,Masses/Density,Standard Norm + Pure Dice

🎓 Academic Reflection
This project demonstrates the application of Software Engineering principles (Modularity, Scalability, and Clean Code) to the field of Medical Artificial Intelligence. By separating the inference logic from the UI, the system remains maintainable and ready for future modality integrations (e.g., MRI or Thermography).