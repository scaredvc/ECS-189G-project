from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Classification import Method_RNN_Classification
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Custom_Train_Test import Setting_Custom_Train_Test
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from script.stage_4_script.tui import get_training_params, print_training_status, print_training_summary
import numpy as np
import torch

data_path = './data/stage_4_data/'

# Get user parameters first
params = get_training_params()
model_type = params['model_type']

print_training_status(f"Running {model_type} model with selected parameters...")

# Handle classification model
if model_type == "classification":
    # Load classification dataset
    dataset = Dataset_Loader(dName="stage 4", dDescription="stage 4", data_type="classification")
    dataset.dataset_source_folder_path = data_path
    
    # Set up classification model and evaluation
    model_obj = Method_RNN_Classification(
        "stage 4 classification", "",  
        max_epoch=int(params['epochs']),  
        learning_rate=float(params['learning_rate'])
    )
    evaluate_obj = Evaluate_Accuracy('accuracy', params['metric'], evaluation_metric=params['metric'])
    setting_obj = Setting_Custom_Train_Test('stage 4 custom train test', '')
    
    # Set up result saver
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result/RNN_Classification_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.file_suffix = f"{params['metric']}"
    
    print_training_status("Classification model configuration complete!", is_complete=True)
    
    # Run classification model
    setting_obj.prepare(sDataset=dataset, sMethod=model_obj, sResult=result_obj, sEvaluate=evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    
    # Display results
    print_training_summary(params['metric'], mean_score, std_score)
    print_training_status("Classification training complete! Model saved successfully.", is_complete=True)

# Handle generation model
elif model_type == "generation":
    # Load generation dataset
    dataset = Dataset_Loader(dName="stage 4 generator", dDescription="joke generator", data_type="generation")
    dataset.dataset_source_folder_path = data_path
    dataset.load()
    
    # Set up generation model
    model_obj = Method_RNN_Generator(
        "stage 4 generator", 
        "RNN for joke generation",
        max_epoch=int(params['epochs']),
        learning_rate=float(params['learning_rate'])
    )
    model_obj.temperature = float(params['temperature'])
    
    # Set up result saver
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result/RNN_Generator_'
    result_obj.result_destination_file_name = 'generation_result'
    
    print_training_status("Generator model configuration complete!", is_complete=True)
    
    # Assign data to the model
    model_obj.data = dataset.data
    
    # Run the generator model
    print_training_status("Training generator model and generating jokes...")
    results = model_obj.run()
    
    # Save results
    result_obj.data = results
    result_obj.save()
    
    # Display sample generations
    print("\n📝 Generated Joke Examples:")
    print("="*60)
    
    for start, joke in results['generated_jokes'].items():
        if start in ["what did the", "why did the", "a man walks"][:3]:
            print(f"Input: \"{start}\"")
            print(f"Generated: \"{joke}\"\n")
    
    print_training_status("Generator training complete! Results saved successfully.", is_complete=True)