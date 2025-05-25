from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import time
import random

class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    
    def __init__(self, mName, mDescription, vocab_size, embedding_dim=100, hidden_dim=256, 
                 num_layers=2, max_epoch=500, learning_rate=1e-3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'Using device: {self.device}')
        
        self.plotting_data = {
            'epoch': [],
            'loss': []
        }
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights properly
        self.init_weights()

    def init_weights(self):
        """Initialize weights for better training"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Embed the input
        embedded = self.embedding(x)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Pass through LSTM
        output, hidden = self.rnn(embedded, hidden)
        
        # Add layer normalization
        output = self.layer_norm(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape output for prediction
        output = output.contiguous().view(-1, self.hidden_dim)
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
    
    def generate_text(self, seed_text, max_length=100, temperature=1.0):
        self.eval()
        tokens = seed_text.split()
        current_ids = torch.LongTensor([self.data['vocab'][t] for t in tokens]).unsqueeze(0)
        current_ids = current_ids.to(self.device)
        
        hidden = None
        generated_tokens = []
        
        with torch.no_grad():
            for i in range(max_length):
                output, hidden = self.forward(current_ids, hidden)
                
                # Get predictions for next word
                predictions = output[:, -1, :] / temperature
                probs = torch.softmax(predictions, dim=-1)
                
                # Sample from the distribution
                next_token_id = torch.multinomial(probs, 1)
                
                generated_tokens.append(next_token_id.item())
                
                # Update input for next iteration
                current_ids = next_token_id.unsqueeze(0)
        
        # Convert token IDs back to words
        rev_vocab = {v: k for k, v in self.data['vocab'].items()}
        generated_text = ' '.join([rev_vocab[i] for i in generated_tokens])
        
        return generated_text
    
    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        
        # Move data to device
        X = X.to(self.device)
        y = y.to(self.device)
        
        for epoch in range(self.max_epoch):
            start_time = time.time()
            
            # Initialize hidden state
            hidden = self.init_hidden(X.size(0))
            
            # Forward pass
            output, hidden = self.forward(X, hidden)
            loss = loss_function(output, y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            optimizer.step()
            
            # Track metrics
            stop_time = time.time()
            time_in_milliseconds = (stop_time - start_time) * 1000
            
            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['loss'].append(loss.item())
            
            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, Time: {time_in_milliseconds:.2f}ms')
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start generation...')
        generated_text = self.generate_text(seed_text="The", max_length=50)
        return {
            'generated_text': generated_text,
            'plotting_data': self.plotting_data
        }