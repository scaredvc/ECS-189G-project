'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle
from local_code.base_class.dataset import dataset
from typing import Literal


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = './data/stage_4_data/'
    
    def __init__(self, dName=None, dDescription=None, data_type: Literal["classification", "generation"] = "classification"):
        super().__init__(dName, dDescription)
        self.data_type = data_type
    
    def load(self):
        if self.data_type == "classification":
            return self._load_classification_data()
        else:
            return self._load_generation_data()
    
    def _load_classification_data(self):
        print('Loading cleaned classification data...')
        with open(os.path.join(self.dataset_source_folder_path, 'cleaned_reviews.pkl'), 'rb') as f:
            reviews, labels = pickle.load(f)  # Data is saved as a tuple of (reviews, labels)
        
        # Split into train (first 80%) and test (last 20%)
        split_idx = int(len(reviews) * 0.8)
        
        result = {
            "training_data": {
                "X": reviews[:split_idx],
                "y": labels[:split_idx]
            },
            "test_data": {
                "X": reviews[split_idx:],
                "y": labels[split_idx:]
            }
        }
        
        self.data = result
        return result
    
    def _load_generation_data(self):
        print('Loading text generation data...')
        with open(os.path.join(self.dataset_source_folder_path, 'text_generation', 'data'), 'r', encoding='utf-8') as f:
            text_data = f.read()
        
        result = {
            "training_data": {"X": text_data, "y": None},
            "test_data": {"X": None, "y": None}
        }
        
        self.data = result
        return result
    
    def get_dimensions(self):
        if self.data_type == "classification":
            # For classification, return average sequence length
            return len(self.data['training_data']['X'][0].split())
        else:
            # For generation, return vocabulary size
            unique_chars = set(self.data['training_data']['X'])
            return len(unique_chars)
    
    def get_output_size(self):
        if self.data_type == "classification":
            return 2  # Binary classification (positive/negative)
        else:
            # For generation, output size is vocabulary size
            unique_chars = set(self.data['training_data']['X'])
            return len(unique_chars)


    # for testing purposes
    def verify_data(self):
        """Verify the loaded classification data"""
        if self.data_type != "classification":
            print("Data verification only available for classification data")
            return
        
        train_data = self.data['training_data']
        test_data = self.data['test_data']
        
        print("\nData Verification:")
        print("=================")
        print("Training Data:")
        print(f"Total examples: {len(train_data['X'])}")
        print(f"Total positive examples: {sum(1 for y in train_data['y'] if y == 1)}")
        print(f"Total negative examples: {sum(1 for y in train_data['y'] if y == 0)}")
        print("\nSample positive review:")
        pos_idx = train_data['y'].index(1)
        print(train_data['X'][pos_idx][:200], "...\n")
        print("Sample negative review:")
        neg_idx = train_data['y'].index(0)
        print(train_data['X'][neg_idx][:200], "...")
        
        print("\nTest Data:")
        print(f"Total examples: {len(test_data['X'])}")
        print(f"Total positive examples: {sum(1 for y in test_data['y'] if y == 1)}")
        print(f"Total negative examples: {sum(1 for y in test_data['y'] if y == 0)}")