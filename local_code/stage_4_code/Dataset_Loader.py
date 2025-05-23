'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import os
from typing import Literal


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    
    def __init__(self, dName=None, dDescription=None, task_type: Literal["text_classification", "text_generation"] = "text_classification"):
        super().__init__(dName, dDescription)
        self.task_type = task_type
    
    def _load_text_classification_data(self):
        result = {
            "training_data": {"X": [], "y": []},
            "test_data": {"X": [], "y": []}
        }
        
        data_splits = {
            "training_data": "train",
            "test_data": "test"
        }
        sentiments = {
            "pos": 1, # Positive label
            "neg": 0  # Negative label
        }

        if self.dataset_source_folder_path is None:
            raise ValueError("dataset_source_folder_path cannot be None for text classification")

        base_path = os.path.join(self.dataset_source_folder_path, "text_classification")

        for split_key, split_folder_name in data_splits.items():
            for sentiment_name, sentiment_label in sentiments.items():
                current_path = os.path.join(base_path, split_folder_name, sentiment_name)
                if not os.path.isdir(current_path):
                    print(f"Warning: Directory not found - {current_path}")
                    continue
                
                for filename in os.listdir(current_path):
                    if filename.endswith(".txt"):
                        file_path = os.path.join(current_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text_content = f.read()
                            result[split_key]["X"].append(text_content)
                            result[split_key]["y"].append(sentiment_label)
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
        return result

    def load(self):
        print(f'loading data for {self.task_type}...')
        if self.task_type == "text_classification":
            return self._load_text_classification_data()
        elif self.task_type == "text_generation":
            # TODO: implement text generation data loading, this is a placeholder
            return {"training_data": {"X": [], "y": []}, "test_data": {"X": [], "y": []}}
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")