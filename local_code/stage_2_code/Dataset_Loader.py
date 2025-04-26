'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None, training_data_file="", test_data_file=""):
        super().__init__(dName, dDescription)
        self.training_data_file = training_data_file
        self.test_data_file = test_data_file
    
    def load(self):
        print('loading data...')
        result = {
            "training_data": {
                "X": [],
                "y": []
            },
            "test_data": {
                "X": [],
                "y": []
            }
        }
        with open(self.dataset_source_folder_path + self.training_data_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.split(',')]
                result["training_data"]["X"].append(elements[1:])
                result["training_data"]["y"].append(elements[0])
        with open(self.dataset_source_folder_path + self.test_data_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.split(',')]
                result["test_data"]["X"].append(elements[1:])
                result["test_data"]["y"].append(elements[0])
        return result