'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import time
from typing import Literal
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader

class Method_CNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, dataset: Dataset_Loader, max_epoch=500, learning_rate=1e-2, input_type: Literal["ORL", "CIFAR", "MNIST"] = "ORL"):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate

        self.input_size = dataset.get_dimensions()
        self.output_size = dataset.get_output_size()

        self.cnn_layer1_conv = nn.Conv2d(in_channels=self.input_size[0], out_channels=1 , kernel_size=3, stride=1, padding=1)
        self.cnn_layer1_relu = nn.ReLU()
        self.cnn_layer1_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.cnn_layer1_dropout = nn.Dropout(p=0.2)

        self.cnn_layer2_conv = nn.Conv2d(in_channels=1 , out_channels=1 , kernel_size=3, stride=1, padding=1)
        self.cnn_layer2_relu = nn.ReLU()
        self.cnn_layer2_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.cnn_layer2_dropout = nn.Dropout(p=0.2)

        self.cnn_layer3_conv = nn.Conv2d(in_channels=1 , out_channels=1 , kernel_size=3, stride=1, padding=1)
        self.cnn_layer3_relu = nn.ReLU()
        self.cnn_layer3_pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.cnn_layer3_dropout = nn.Dropout(p=0.2)
        
        self.flatten = nn.Flatten()

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

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, input_features):
        '''Forward propagation'''

        # # First hidden layer
        # hidden1 = self.fc_layer_1(input_features)
        # hidden1_norm = self.batch_norm_1(hidden1)
        # hidden1_activated = self.activation_func_1(hidden1_norm)
        # hidden1_regularized = self.dropout_1(hidden1_activated)

        # # Second hidden layer
        # hidden2 = self.fc_layer_2(hidden1_regularized)
        # hidden2_norm = self.batch_norm_2(hidden2)
        # hidden2_activated = self.activation_func_2(hidden2_norm)
        # hidden2_regularized = self.dropout_2(hidden2_activated)

        # # Third hidden layer
        # hidden3 = self.fc_layer_3(hidden2_regularized)
        # hidden3_norm = self.batch_norm_3(hidden3)
        # hidden3_activated = self.activation_func_3(hidden3_norm)
        # hidden3_regularized = self.dropout_3(hidden3_activated)

        # # Output layer
        # logits = self.fc_output_layer(hidden3_regularized)

        # return logits

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            start_time = time.time()

            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(torch.FloatTensor(X).to(self.device))
            # convert y to torch.tensor as well
            y_true = torch.LongTensor(np.array(y)).to(self.device)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            accuracy_evaluator.data = {
                'true_y': y_true.cpu(),
                'pred_y': y_pred.max(1)[1].cpu()
            }

            stop_time = time.time()
            time_in_milliseconds = (stop_time - start_time) * 1000

            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['loss'].append(train_loss.item())
            print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item(), 'Time:', str(time_in_milliseconds) + 'ms')
    
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(self.device))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1].cpu()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], "plotting_data": self.plotting_data}
            