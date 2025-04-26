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
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
for idx, (metric, label) in enumerate(metrics):
    loader = Result_Loader('result loader', '')
    loader.result_destination_folder_path = result_folder
    loader.result_destination_file_name = f'MLP_prediction_result_1_{metric}'
    loader.load()
    data = loader.data
    plotting_data = data['plotting_data']
    ax = axes[idx // 2, idx % 2]
    ax.plot(plotting_data['epoch'], plotting_data['loss'], label=label, color=colors[idx])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{label}')
    ax.grid(True)
    ax.legend()
fig.suptitle('Training Loss vs. Epoch (MLP Convergence by Metric)', fontsize=18)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()





