from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
import sys
import os

# this script tests that our dataset loader is working correctly

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

def test_text_classification_loader():
    print("--- Testing Text Classification Dataset Loader ---")
    
    data_source_path = os.path.join(project_root, "data", "stage_4_data")
    
    data_loader = Dataset_Loader(dName="Movie Reviews", dDescription="Sentiment classification dataset", task_type="text_classification")
    data_loader.dataset_source_folder_path = data_source_path

    loaded_data = data_loader.load()

    if not loaded_data or not loaded_data.get('training_data', {}).get('X') and not loaded_data.get('test_data', {}).get('X'):
        print("Failed to load data or data is empty.")
        if hasattr(data_loader, 'dataset_source_folder_path'):
            print(f"Attempted to load from: {data_loader.dataset_source_folder_path}")
            text_classification_path = os.path.join(data_loader.dataset_source_folder_path, 'text_classification')
            print(f"Checking existence of text_classification dir: {text_classification_path} - Exists: {os.path.isdir(text_classification_path)}")
            if os.path.isdir(text_classification_path):
                train_pos_path = os.path.join(text_classification_path, 'train', 'pos')
                print(f"Checking existence of train/pos dir: {train_pos_path} - Exists: {os.path.isdir(train_pos_path)}")
        return

    print(f"Training samples: {len(loaded_data['training_data']['X'])}")
    print(f"Test samples: {len(loaded_data['test_data']['X'])}")

    print("\\n--- First 2 Training Samples ---")
    for i in range(min(2, len(loaded_data['training_data']['X']))):
        print(f"Sample {i+1}:")
        print(f"  Text: {loaded_data['training_data']['X'][i][:200]}...") # Print first 200 chars
        print(f"  Label: {loaded_data['training_data']['y'][i]}")

    print("\\n--- First 2 Test Samples ---")
    for i in range(min(2, len(loaded_data['test_data']['X']))):
        print(f"Sample {i+1}:")
        print(f"  Text: {loaded_data['test_data']['X'][i][:200]}...") # Print first 200 chars
        print(f"  Label: {loaded_data['test_data']['y'][i]}")

    print("\\n--- Test Complete ---")

if __name__ == '__main__':
    test_text_classification_loader()
