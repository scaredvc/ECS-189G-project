from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

data_path = './data/stage_2_data/'

training_data = Dataset_Loader('stage 2 data train', '')
training_data.dataset_source_folder_path = data_path
training_data.dataset_source_file_name = 'train.csv'

test_data = Dataset_Loader('stage 2 data test', '')
test_data.dataset_source_folder_path = data_path
test_data.dataset_source_file_name = 'test.csv'

model_obj = Method_MLP("stage 2 method", "")

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = './result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'

setting_obj = Setting_KFold_CV('stage 2 k fold cross validation', '')

evaluate_obj = Evaluate_Accuracy('accuracy', 'accuracy_score', evaluation_metric="accuracy_score")

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(training_data, model_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
mean_score, std_score = setting_obj.load_run_save_evaluate()
print('************ Overall Performance ************')
print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
print('************ Finish ************')
# ------------------------------------------------------
    
