from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Generation import Method_RNN_Generator
from local_code.stage_4_code.Result_Saver import Result_Saver
# from local_code.stage_4_code.Setting_Custom_Train_Test import Setting_Custom_Train_Test # Assuming not used
# from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy # Assuming not used for generation
# from script.stage_4_script.tui import get_training_params, print_training_status, print_training_summary

import os # Import os
import json

# data_path = './data/stage_4_data/' # This was not used effectively

def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Create necessary directories
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'stage_4_data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Load the preprocessed jokes from JSON
    json_path = os.path.join(data_dir, 'cleaned_generation.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        joke_data = json.load(f)
    
    # Initialize the model
    dataset = Dataset_Loader('text_generation', 'Dataset for text generation')
    dataset.dataset_source_folder_path = data_dir
    
    # Set the data directly
    dataset.data = {
        'train': {
            'X': joke_data['train'],  # These should be the preprocessed jokes
            'y': None  # Not needed for generation
        },
        'test': {
            'X': joke_data['test'],
            'y': None
        }
    }
    
    method = Method_RNN_Generator('transformer_generator', 'Transformer for joke generation')
    method.data = dataset.data
    
    # Rest of your code...

if __name__ == '__main__':
    main()