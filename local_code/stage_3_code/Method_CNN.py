'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import time
from typing import Literal
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from torch.utils.data import TensorDataset, DataLoader

class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2
    # batch size for mini-batch training
    batch_size = 128

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, dataset: Dataset_Loader, max_epoch=500, learning_rate=1e-2, input_type: Literal["ORL", "CIFAR", "MNIST"] = "ORL", batch_size=128):
        # ORL dataset is small, use full batch for it
        self.use_mini_batch = input_type != "ORL"
        self.input_type = input_type
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Determine input dimensions and ensure correct channel order (C, H, W)
        raw_dim = dataset.get_dimensions()
        if len(raw_dim) == 2:
            # grayscale image: (H, W) -> (1, H, W)
            channels, height, width = 1, raw_dim[0], raw_dim[1]
        elif len(raw_dim) == 3:
            # Could be (H, W, C) or (C, H, W). Assume channel-last if last dim is 1/3
            if raw_dim[2] in {1, 3} and raw_dim[0] not in {1, 3}:
                channels, height, width = raw_dim[2], raw_dim[0], raw_dim[1]
            else:
                channels, height, width = raw_dim[0], raw_dim[1], raw_dim[2]
        else:
            raise ValueError(f"Unsupported input dimension format: {raw_dim}")

        self.input_size = (channels, height, width)
        self.output_size: int = dataset.get_output_size()

        # Add more padding for ORL dataset or reduce pooling operations
        self.cnn_layer1_conv = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cnn_layer1_relu = nn.ReLU()
        self.cnn_layer1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Use moderate dropout
        self.cnn_layer1_dropout = nn.Dropout(p=0.3)

        self.cnn_layer2_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.cnn_layer2_relu = nn.ReLU()
        # Check if dimensions are too small for a second pooling layer
        h, w = height, width
        h_after_pool1 = h // 2
        w_after_pool1 = w // 2
        
        # Only use second pooling if dimensions allow it
        if h_after_pool1 >= 4 and w_after_pool1 >= 4:
            self.cnn_layer2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
            h_out = h_after_pool1 // 2
            w_out = w_after_pool1 // 2
        else:
            # Use a smaller pooling or no pooling for the second layer
            self.cnn_layer2_pool = nn.Identity()  # No pooling
            h_out = h_after_pool1
            w_out = w_after_pool1
            
        self.cnn_layer2_dropout = nn.Dropout(p=0.3)
        
        self.flatten = nn.Flatten()

        # No need to calculate sizes again - they're calculated above
        
        # Print dimensions for debugging
        print(f"Input dimensions (C,H,W): {self.input_size}, Output after processing: {h_out}x{w_out}")
        
        # Adjust fully connected layers with the calculated dimensions
        self.fc_layer1 = nn.Linear(in_features=64 * h_out * w_out, out_features=128)
        self.fc_relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(p=0.3)
        self.fc_layer2 = nn.Linear(in_features=128, out_features=self.output_size)

        # Move model to GPU if available
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA device count: {torch.cuda.device_count()}')
            print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'Using device: {self.device}')

        self.plotting_data = {
            'epoch': [],
            'loss': []
        }

    def _prepare_input(self, X):
        """Convert numpy/list image batch to torch tensor with shape (N, C, H, W)."""
        X_tensor = torch.FloatTensor(np.array(X))  # ensure contiguous tensor
        
        # Handle different dimension formats
        if X_tensor.ndim == 3:
            # (N, H, W) -> (N, 1, H, W)
            X_tensor = X_tensor.unsqueeze(1)
        elif X_tensor.ndim == 4:
            # if channel last (N, H, W, C) swap to channel first
            if X_tensor.shape[-1] == self.input_size[0]:
                X_tensor = X_tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError("Unexpected input tensor dimensions. Expected 3 or 4 dims.")
            
        # Simple normalization to [0,1] range
        if X_tensor.max() > 1.0:
            X_tensor = X_tensor / 255.0
            
        return X_tensor.to(self.device)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, input_features):
        '''Forward propagation'''
        layer1_conv = self.cnn_layer1_conv(input_features)
        layer1_relu = self.cnn_layer1_relu(layer1_conv)
        layer1_pool = self.cnn_layer1_pool(layer1_relu)
        layer1_dropout = self.cnn_layer1_dropout(layer1_pool)

        layer2_conv = self.cnn_layer2_conv(layer1_dropout)
        layer2_relu = self.cnn_layer2_relu(layer2_conv)
        layer2_pool = self.cnn_layer2_pool(layer2_relu)
        layer2_dropout = self.cnn_layer2_dropout(layer2_pool)

        flatten = self.flatten(layer2_dropout)
        
        fc_layer1 = self.fc_layer1(flatten)
        fc_relu = self.fc_relu(fc_layer1)
        fc_dropout = self.fc_dropout(fc_relu)
        fc_layer2 = self.fc_layer2(fc_dropout)
        
        # Don't use softmax here since CrossEntropyLoss applies it internally
        return fc_layer2

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def fit_model(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        # Use Adam optimizer with a small weight decay
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # Standard cross entropy loss
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        # Preprocess data once (not in every epoch)
        X_tensor = self._prepare_input(X)
        y_tensor = torch.LongTensor(np.array(y)).to(self.device)
        
        # For ORL use full-batch, for MNIST use mini-batch
        if self.use_mini_batch:
            # Create DataLoader for mini-batch training
            train_dataset = TensorDataset(X_tensor, y_tensor)
            dataset_size = len(train_dataset)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            # Full batch training (better for small datasets like ORL)
            dataset_size = len(X_tensor)  # Size of the dataset
            train_loader = [(X_tensor, y_tensor)]  # Single batch with all data
        
        # Training loop
        for epoch in range(self.max_epoch):
            start_time = time.time()
            self.train()  # Set model to training mode
            epoch_loss = 0
            all_preds = []
            all_true = []
            
            # Mini-batch training
            for batch_x, batch_y in train_loader:
                # Forward pass
                pred = self.forward(batch_x)
                loss = loss_function(pred, batch_y)
                    
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Collect data for accuracy calculation
                epoch_loss += loss.item() * batch_x.size(0)
                all_preds.append(pred.max(1)[1].detach().cpu())
                all_true.append(batch_y.cpu())
            
            # Calculate epoch statistics
            epoch_loss /= dataset_size
            all_preds = torch.cat(all_preds)
            all_true = torch.cat(all_true)
            
            accuracy_evaluator.data = {
                'true_y': all_true,
                'pred_y': all_preds
            }
            accuracy = accuracy_evaluator.evaluate()

            stop_time = time.time()
            time_in_milliseconds = (stop_time - start_time) * 1000

            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['loss'].append(epoch_loss)
            print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {epoch_loss:.6f}, Time: {time_in_milliseconds:.0f}ms')
    
    def test(self, X):
        # do the testing, and result the result
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            # For ORL, we can process all at once
            if not self.use_mini_batch or len(X) <= self.batch_size:
                y_pred = self.forward(self._prepare_input(X))
                return y_pred.max(1)[1].cpu()
            # For larger datasets like MNIST, process in batches
            else:
                all_preds = []
                X_tensor = self._prepare_input(X)
                # Create test DataLoader
                test_dataset = TensorDataset(X_tensor)
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
                
                for batch in test_loader:
                    batch_x = batch[0]
                    batch_pred = self.forward(batch_x)
                    all_preds.append(batch_pred.max(1)[1].cpu())
                
                return torch.cat(all_preds)
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.fit_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], "plotting_data": self.plotting_data}