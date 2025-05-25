from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_DT import Method_DT
from local_code.stage_4_code.Method_SVM import Method_SVM
from local_code.stage_4_code.Result_Saver import Result_Saver
from local_code.stage_4_code.Setting_Custom_Train_Test import Setting_Custom_Train_Test
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
# from script.stage_4_script.tui import get_training_params, print_training_status, print_training_summary

data_path = './data/stage_4_data/'

# dataset = Dataset_Loader('stage 4 data train', '', training_data_file='train.csv', test_data_file='test.csv')
# dataset.dataset_source_folder_path = data_path

# params = get_training_params()

dataset = Dataset_Loader("Text Classification", "Movie Reviews", data_type="classification")
dataset.load()
dataset.verify_data()