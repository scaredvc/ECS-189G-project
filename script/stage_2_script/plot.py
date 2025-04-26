from local_code.stage_2_code.Result_Loader import Result_Loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

loader = Result_Loader('result loader', '')
loader.result_destination_folder_path = './result/stage_2_result/'
loader.result_destination_file_name = 'MLP_prediction_result'
loader.fold_count = 1  # or 2, or 3

loader.load()

pred_y = loader.data['pred_y']
true_y = loader.data['true_y']

if hasattr(pred_y, 'cpu'):
    pred_y = pred_y.cpu().numpy()

cm = confusion_matrix(true_y, pred_y)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix')
plt.show()

plt.figure()
plt.hist(true_y, bins=np.arange(true_y.min(), true_y.max()+2)-0.5, alpha=0.5, label='True')
plt.hist(pred_y, bins=np.arange(pred_y.min(), pred_y.max()+2)-0.5, alpha=0.5, label='Predicted')
plt.legend()
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('True vs Predicted Label Distribution')
plt.show()

print(loader.data['pred_y'])