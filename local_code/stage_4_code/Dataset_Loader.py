'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import json
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
        with open(os.path.join(self.dataset_source_folder_path, 'cleaned_classification.json'), 'r', encoding="utf-8") as f:
            data = json.load(f)

        training = data["train"]
        testing = data["test"]

        pos_train = training["pos"]
        neg_train = training["neg"]

        pos_test = testing["pos"]
        neg_test = testing["neg"]

        result = {
            "training_data": {
                "X": pos_train + neg_train,
                "y": [1 for _ in pos_train] + [0 for _ in neg_train]
            },
            "test_data": {
                "X": pos_test + neg_test,
                "y": [1 for _ in pos_test] + [0 for _ in neg_test]
            }
        }

        self.data = result
        return result
    
    def _load_generation_data(self):
        print('Loading text generation data...')
        import pandas as pd
        import os
        
        # Load the jokes from CSV file
        csv_path = os.path.join(self.dataset_source_folder_path, 'cleaned_generation.csv')
        df = pd.read_csv(csv_path)
        
        # Extract jokes text, ignoring the ID column
        jokes = df['Joke'].tolist()
        
        # For text generation, we don't need to split into train/test
        # We'll use all data for training the generator
        result = {
            'jokes': jokes,
            # We'll also create a list of jokes with their first five words as input
            # and the rest as target for training the generator
            'joke_inputs': [],
            'joke_targets': []
        }
        
        # Process each joke to create input/target pairs
        for joke in jokes:
            words = joke.lower().split()
            if len(words) > 5:  # Only use jokes with more than 5 words
                input_words = ' '.join(words[:5])
                target_words = ' '.join(words[5:])
                result['joke_inputs'].append(input_words)
                result['joke_targets'].append(target_words)
        
        print(f"Loaded {len(jokes)} jokes, {len(result['joke_inputs'])} usable for training")
        self.data = result
        return result
    
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