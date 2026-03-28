import os
from pathlib import Path

def create_project_structure(base_path):
    # Define the folder hierarchy
    folders = [
        "data/raw",
        "data/processed/train/images",
        "data/processed/train/masks",
        "data/processed/test/images",
        "data/processed/test/masks",
        "models",
        "notebooks",
        "src",
    ]
    
    # Define initial empty files
    files = [
        "requirements.txt",
        "README.md",
        "src/model_builder.py",
        "src/patientFetcher.py",
        "app.py",
        ".gitignore"
    ]

    base = Path(base_path)

    # Create folders
    for folder in folders:
        folder_path = base / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {folder_path}")

    # Create empty files
    for file in files:
        file_path = base / file
        if not file_path.exists():
            file_path.touch()
            print(f"Created file: {file_path}")
            
            # Optional: Add a starter note to the README
            if file == "README.md":
                file_path.write_text("# Breast Cancer Tracker\nStudent Project for Longitudinal Analysis.")
            
            # Optional: Pre-fill .gitignore to avoid uploading huge medical data
            if file == ".gitignore":
                file_path.write_text("data/\n__pycache__/\n*.pth\n.ipynb_checkpoints/")

    print("\n✅ Project structure is ready!")

if __name__ == "__main__":
    # Change '.' to a specific path if you want to build it elsewhere
    # e.g., r"D:\Projet Federateur\AI\BCT"
    target_path = r"D:\Projet Fédérateur\AI\BCT"
    create_project_structure(target_path)