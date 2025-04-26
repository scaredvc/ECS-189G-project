from local_code.stage_2_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot training loss vs. epoch for all four metrics
from local_code.stage_2_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
import numpy as np

result_folder = './result/stage_2_result/'
metrics = [
    ('accuracy_score', 'Accuracy'),
    ('f1_score', 'F1 Score'),
    ('precision_score', 'Precision'),
    ('recall_score', 'Recall')
]

colors = ['blue', 'green', 'red', 'orange']
for idx, (metric, label) in enumerate(metrics):
    loader = Result_Loader('result loader', '')
    loader.result_destination_folder_path = result_folder
    loader.result_destination_file_name = f'MLP_prediction_result_1_{metric}'
    loader.load()
    data = loader.data
    plotting_data = data['plotting_data']
    plt.figure(figsize=(8, 5))
    plt.plot(plotting_data['epoch'], plotting_data['loss'], label=label, color=colors[idx])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss vs. Epoch ({label})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



