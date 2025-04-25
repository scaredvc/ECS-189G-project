from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

data_path = './data/stage_2_data/'

training_data = Dataset_Loader('stage2data_train', '')
training_data.dataset_source_folder_path = data_path
training_data.dataset_source_file_name = 'train.csv'
training_data = training_data.load()

test_data = Dataset_Loader('stage2data_test', '')
test_data.dataset_source_folder_path = data_path
test_data.dataset_source_file_name = 'test.csv'
test_data = test_data.load()

print(training_data)
print(test_data)