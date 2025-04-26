from local_code.stage_2_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

loader = Result_Loader('result loader', '')
loader.result_destination_folder_path = './result/stage_2_result/'
loader.result_destination_file_name = 'MLP_prediction_result_1_accuracy_score'

loader.load()

data = loader.data

plotting_data = data['plotting_data']

# Plot training loss vs. epoch
plt.figure(figsize=(8, 5))
plt.plot(plotting_data['epoch'], plotting_data['loss'], label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch (MLP Convergence)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


