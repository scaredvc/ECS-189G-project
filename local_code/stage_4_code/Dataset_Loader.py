'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import json
import pandas as pd
from local_code.base_class.dataset import dataset
from typing import Literal


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = './data/stage_4_data/'

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        # The dataset_name from dName will determine which loader to use
        self.dataset_source_folder_path = os.path.join(self.dataset_source_folder_path, 
                                                      'text_generation' if dName == 'text_generation' else '')

    def load(self):
        """Load data based on dataset name"""
        if self.dataset_name == 'text_generation':
            return self._load_generation_data()
        else:
            return self._load_classification_data()

    def _load_generation_data(self):
        """Load data for text generation task"""
        try:
            target_file_path = os.path.join(self.dataset_source_folder_path, 'cleaned_generation.csv')
            print(f"Dataset_Loader: Loading data from: {target_file_path}")
            
            df = pd.read_csv(target_file_path)
            print(f"Available columns in CSV: {df.columns.tolist()}")  # Debug print
            
            # Try different possible column names
            if 'Joke' in df.columns:
                texts = df['Joke'].tolist()
            elif 'joke' in df.columns:
                texts = df['joke'].tolist()
            elif 'text' in df.columns:
                texts = df['text'].tolist()
            else:
                print(f"Error: No suitable text column found. Available columns: {df.columns.tolist()}")
                return None
            
            # Split into train/valid/test
            train_size = int(0.8 * len(texts))
            valid_size = int(0.1 * len(texts))
            
            print(f"Dataset_Loader: Successfully loaded {len(texts)} texts")
            return {
                'train': {'X': texts[:train_size]},
                'valid': {'X': texts[train_size:train_size + valid_size]},
                'test': {'X': texts[train_size + valid_size:]}
            }
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in directory: {os.listdir(self.dataset_source_folder_path)}")
            return None

    def _load_classification_data(self):
        """Load data for classification task"""
        with open(os.path.join(self.dataset_source_folder_path, 'cleaned_classification.json'), 'r', encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    # for testing purposes
    def verify_data(self):
        """Verify the loaded classification data"""
        if self.dataset_name != "classification":
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