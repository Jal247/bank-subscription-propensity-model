import os
import sys

def execute_project():
    print("🔔 Starting Full Bank Marketing Project Workflow...")

    # 1. Run Preprocessing
    print("\n--- STEP 1: PREPROCESSING ---")
    # This calls your preprocess_data.py script
    exit_code_1 = os.system('python src/preprocessing.py')
    
    if exit_code_1 != 0:
        print("❌ Error occurred in Preprocessing. Stopping pipeline.")
        sys.exit(1)
    
    print("✅ Preprocessing Successful.")

    # 2. Run Training & Evaluation
    print("\n--- STEP 2: MODEL TRAINING & EVALUATION ---")
    # This calls your pipeline.py script
    exit_code_2 = os.system('python src/pipelines.py')
    
    if exit_code_2 != 0:
        print("❌ Error occurred in Pipeline. Stopping.")
        sys.exit(1)

    print("\n🎉 Full Project Execution Finished Successfully!")
    print("📍 Check the /models folder for artifacts and /image for plots.")

if __name__ == "__main__":
    execute_project()
