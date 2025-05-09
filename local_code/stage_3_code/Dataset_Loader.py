'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
from local_code.base_class.dataset import dataset
from typing import Literal

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None, data_file: Literal["ORL", "CIFAR", "MNIST"] = "ORL"):
        super().__init__(dName, dDescription)
        self.data_file = data_file
    
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
        with open(self.dataset_source_folder_path + self.data_file, 'rb') as f:
            self.data = pickle.load(f)
            for instance in self.data['train']:
                image_matrix = instance['image']
                image_label = instance['label']
                result["training_data"]["X"].append(image_matrix)
                result["training_data"]["y"].append(image_label)
            for instance in self.data['test']:
                image_matrix = instance['image']
                image_label = instance['label']
                result["test_data"]["X"].append(image_matrix)
                result["test_data"]["y"].append(image_label)
        return result
    
    def get_dimensions(self):
        return self.data['train'][0]['image'].shape
