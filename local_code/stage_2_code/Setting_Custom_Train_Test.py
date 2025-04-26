'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_Custom_Train_Test(setting):

    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
    
    def load_run_save_evaluate(self):
        # load dataset
        loaded_data = self.dataset.load()

        X = np.array(loaded_data["training_data"]["X"])
        y = np.array(loaded_data["training_data"]["y"])
        test_X = np.array(loaded_data["test_data"]["X"])
        test_y = np.array(loaded_data["test_data"]["y"])

        # Train on the entire training set and evaluate on the entire test set
        self.method.data = {'train': {'X': X, 'y': y}, 'test': {'X': test_X, 'y': test_y}}
        learned_result = self.method.run()

        # Save result
        self.result.data = learned_result
        self.result.fold_count = 1
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        return score, None

        