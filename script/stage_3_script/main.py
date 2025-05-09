from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_Custom_Train_Test import Setting_Custom_Train_Test
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from script.stage_3_script.tui import get_training_params, print_training_status, print_training_summary
import numpy as np
import torch

data_path = './data/stage_3_data/'

dataset = Dataset_Loader('stage 3 data train', '', data_file='ORL')
dataset.dataset_source_folder_path = data_path

params = get_training_params()
dataset.load()
print_training_status("Creating model with selected parameters...")
model_obj = Method_CNN("stage 3 method", "",dataset=dataset,  max_epoch=int(params['epochs']),  learning_rate=float(params['learning_rate']))
evaluate_obj = Evaluate_Accuracy('accuracy', params['metric'], evaluation_metric=params['metric'])
setting_obj = Setting_Custom_Train_Test('stage 3 custom train test', '')
result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = './result/stage_3_result/CNN_'
result_obj.result_destination_file_name = 'prediction_result'
result_obj.file_suffix = f"{params['metric']}"
print_training_status("Configuration complete!", is_complete=True)

setting_obj.prepare(sDataset=dataset, sMethod=model_obj, sResult=result_obj, sEvaluate=evaluate_obj)
setting_obj.print_setup_summary()
mean_score, std_score = setting_obj.load_run_save_evaluate()

print_training_summary(params['metric'], mean_score, std_score)
print_training_status("Training complete! Model saved successfully.", is_complete=True)