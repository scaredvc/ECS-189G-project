'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Literal


class Evaluate_Accuracy(evaluate):
    data = None
    
    def __init__(self, eName=None, eDescription=None, evaluation_metric: Literal["accuracy_score", "f1_score", "precision_score", "recall_score"] = "accuracy_score"):
        super().__init__(eName, eDescription)
        self.evaluate_name = evaluation_metric

    def evaluate(self):
        print('evaluating performance...')
        match self.evaluate_name:
            case "accuracy_score":
                return accuracy_score(self.data['true_y'], self.data['pred_y'])
            case "f1_score":
                return f1_score(self.data['true_y'], self.data['pred_y'], average='weighted')
            case "precision_score":
                return precision_score(self.data['true_y'], self.data['pred_y'], average='weighted')
            case "recall_score":
                return recall_score(self.data['true_y'], self.data['pred_y'], average='weighted')
            case _:
                raise ValueError("Unknown evaluation metric")