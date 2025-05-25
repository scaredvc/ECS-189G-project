'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD


from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    
    def __init__(self, mName, mDescription, vocab_size, embedding_dim=100, hidden_dim=128, 
                 num_layers=2, max_epoch=500, learning_rate=1e-3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.2)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'Using device: {self.device}')
        
        self.plotting_data = {
            'epoch': [],
            'loss': []
        }

    def forward(self, text, lengths):
        # text shape: (batch_size, seq_length)
        
        # Embed the text
        embedded = self.embedding(text)  # (batch_size, seq_length, embedding_dim)
        
        # Pack padded sequence for RNN efficiency
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through RNN
        packed_output, (hidden, cell) = self.rnn(packed_input)
        
        # Use the final hidden state from both directions
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        # Pass through linear layer
        output = self.fc(hidden)
        
        return output

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        # Convert data to tensors
        X_lengths = torch.LongTensor([len(x) for x in X])
        X_padded = pad_sequence([torch.LongTensor(x) for x in X], batch_first=True)
        y = torch.LongTensor(np.array(y))
        
        # Move to device
        X_padded = X_padded.to(self.device)
        X_lengths = X_lengths.to(self.device)
        y = y.to(self.device)
        
        for epoch in range(self.max_epoch):
            start_time = time.time()
            
            # Forward pass
            y_pred = self.forward(X_padded, X_lengths)
            
            # Calculate loss
            train_loss = loss_function(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # Evaluate accuracy
            accuracy_evaluator.data = {
                'true_y': y.cpu(),
                'pred_y': y_pred.max(1)[1].cpu()
            }
            
            # Track metrics
            stop_time = time.time()
            time_in_milliseconds = (stop_time - start_time) * 1000
            
            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['loss'].append(train_loss.item())
            
            print('Epoch:', epoch, 
                  'Accuracy:', accuracy_evaluator.evaluate(), 
                  'Loss:', train_loss.item(), 
                  'Time:', str(time_in_milliseconds) + 'ms')
    
    def test(self, X):
        self.eval()
        with torch.no_grad():
            X_lengths = torch.LongTensor([len(x) for x in X])
            X_padded = pad_sequence([torch.LongTensor(x) for x in X], batch_first=True)
            
            X_padded = X_padded.to(self.device)
            X_lengths = X_lengths.to(self.device)
            
            outputs = self.forward(X_padded, X_lengths)
            return outputs.max(1)[1].cpu()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], "plotting_data": self.plotting_data}