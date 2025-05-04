from local_code.stage_3_code.Dataset_Loader import Dataset_Loader

# test data loader
data_path = './data/stage_3_data/'

dataset = Dataset_Loader('stage 3 data train', '', data_file='ORL')
dataset.dataset_source_folder_path = data_path

result = dataset.load()
print(result)