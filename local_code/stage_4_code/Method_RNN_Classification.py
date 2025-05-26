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
from collections import Counter
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


class Method_RNN_Classification(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    word_to_idx = {}
    idx_to_word = {}
    vocab_size = 0
    max_seq_length = 200
    embedding_dim = 100 
    hidden_dim = 64
    num_layers = 2
    bidirectional = True
    dropout_rate = 0.2 

    def __init__(self, mName, mDescription, max_epoch=500, learning_rate=1e-3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Store parameters
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        
        # Initialize other attributes
        self.plotting_data = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': []
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA device count: {torch.cuda.device_count()}')
            print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'Using device: {self.device}')

    def _build_vocab(self, texts, min_freq=2):
        """Build vocabulary from texts"""
        print('Building vocabulary...')
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            # Split by whitespace
            words = text.lower().split()
            word_counts.update(words)
        
        # Filter by minimum frequency
        filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Create mappings
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, word in enumerate(filtered_words, start=2):
            self.word_to_idx[word] = idx
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f'Vocabulary size: {self.vocab_size}')
        return self.word_to_idx, self.idx_to_word, self.vocab_size

    def _text_to_indices(self, texts):
        """Convert texts to sequences of indices"""
        sequences = []
        sequence_lengths = []
        
        for text in texts:
            words = text.lower().split()
            # Truncate if longer than max_seq_length
            if len(words) > self.max_seq_length:
                words = words[:self.max_seq_length]
            
            # Convert to indices
            indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            sequences.append(indices)
            sequence_lengths.append(len(indices))
            
        return sequences, sequence_lengths

    def _build_model(self):
        """Initialize model layers after vocabulary is built"""
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Output layer
        output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(output_dim, 2)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Move model to device
        self.to(self.device)

    def forward(self, text, lengths):
        """Forward propagation"""
        # Embed the text - (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(text)
        
        # Pack padded sequence for RNN efficiency
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # Pass through RNN
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        
        # Get the final hidden state from both directions
        if self.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply dropout to the final hidden state
        hidden = self.dropout(hidden)
        
        # Pass through the fully connected layer
        output = self.fc(hidden)
        
        return output

    def train_model(self, X, y):
        """Train the model using mini-batches for memory efficiency"""
        # First build vocabulary if not done yet
        if self.vocab_size == 0:
            self._build_vocab(X)
            self._build_model()
        
        # Convert texts to sequences of indices
        X_indices, X_lengths = self._text_to_indices(X)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        # Define batch size
        batch_size = 128  # Adjust this based on your GPU memory
        
        # Create dataset indices for batching
        num_samples = len(X_indices)
        indices = list(range(num_samples))
        
        # Training loop
        for epoch in range(self.max_epoch):
            start_time = time.time()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            # Set model to training mode
            self.train(True)
            
            # Shuffle indices each epoch for randomized batches
            np.random.shuffle(indices)
            
            # Process mini-batches
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                batch_X = [X_indices[i] for i in batch_indices]
                batch_lengths = [X_lengths[i] for i in batch_indices]
                batch_y = [y[i] for i in batch_indices]
                
                # Convert to tensors and pad sequences
                batch_X_tensor = [torch.LongTensor(x) for x in batch_X]
                batch_X_padded = pad_sequence(batch_X_tensor, batch_first=True, padding_value=0)
                batch_lengths_tensor = torch.LongTensor(batch_lengths)
                batch_y_tensor = torch.LongTensor(batch_y)
                
                # Move data to device - lengths must stay on CPU
                batch_X_padded = batch_X_padded.to(self.device)
                batch_y_tensor = batch_y_tensor.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                batch_pred = self.forward(batch_X_padded, batch_lengths_tensor)
                
                # Calculate loss
                batch_loss = loss_function(batch_pred, batch_y_tensor)
                epoch_loss += batch_loss.item() * len(batch_indices)
                
                # Backward pass and optimization
                batch_loss.backward()
                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Calculate accuracy stats using batch predictions
                batch_predicted = batch_pred.argmax(dim=1)
            
            # Collect all predictions and targets for accuracy evaluation
            all_preds = []
            all_targets = []
            
            # Set model to evaluation mode for accuracy assessment
            self.eval()
            
            with torch.no_grad():
                for start_idx in range(0, num_samples, batch_size):
                    batch_indices = indices[start_idx:start_idx + batch_size]
                    
                    # Get batch data
                    batch_X = [X_indices[i] for i in batch_indices]
                    batch_lengths = [X_lengths[i] for i in batch_indices]
                    batch_y = [y[i] for i in batch_indices]
                    
                    # Convert to tensors and pad sequences
                    batch_X_tensor = [torch.LongTensor(x) for x in batch_X]
                    batch_X_padded = pad_sequence(batch_X_tensor, batch_first=True, padding_value=0)
                    batch_lengths_tensor = torch.LongTensor(batch_lengths)
                    batch_y_tensor = torch.LongTensor(batch_y)
                    
                    # Move data to device - lengths must stay on CPU
                    batch_X_padded = batch_X_padded.to(self.device)
                    batch_y_tensor = batch_y_tensor.to(self.device)
                    
                    # Forward pass
                    batch_pred = self.forward(batch_X_padded, batch_lengths_tensor)
                    
                    # Get predicted labels
                    batch_predicted = batch_pred.argmax(dim=1).cpu()
                    all_preds.append(batch_predicted)
                    all_targets.append(batch_y_tensor.cpu())
            
            # Set model back to training mode
            self.train(True)
            
            # Concatenate all predictions and targets
            all_predictions = torch.cat(all_preds)
            all_targets = torch.cat(all_targets)
            
            # Use Evaluate_Accuracy for evaluation
            accuracy_evaluator.data = {
                'true_y': all_targets,
                'pred_y': all_predictions
            }
            accuracy = accuracy_evaluator.evaluate()
            
            # Calculate average loss
            avg_loss = epoch_loss / num_samples
            
            # Store metrics for plotting
            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['train_loss'].append(avg_loss)
            self.plotting_data['train_accuracy'].append(accuracy)
            
            # Print progress
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # convert to milliseconds
            print(f'Epoch: {epoch}/{self.max_epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Time: {elapsed_time:.2f}ms')
    
    def test(self, X):
        """Test the model on new data using mini-batches"""
        # Set model to evaluation mode
        self.eval()
        
        # Convert texts to sequences of indices
        X_indices, X_lengths = self._text_to_indices(X)
        
        # Define batch size for testing
        batch_size = 128  # Same as training for consistency
        num_samples = len(X_indices)
        
        # Prepare to collect predictions
        all_predictions = []
        
        with torch.no_grad():
            # Process mini-batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = list(range(start_idx, end_idx))
                
                # Get batch data
                batch_X = [X_indices[i] for i in batch_indices]
                batch_lengths = [X_lengths[i] for i in batch_indices]
                
                # Convert to tensors and pad sequences
                batch_X_tensor = [torch.LongTensor(x) for x in batch_X]
                batch_X_padded = pad_sequence(batch_X_tensor, batch_first=True, padding_value=0)
                batch_lengths_tensor = torch.LongTensor(batch_lengths)
                
                # Move data to device - lengths must stay on CPU
                batch_X_padded = batch_X_padded.to(self.device)
                
                # Forward pass
                batch_pred = self.forward(batch_X_padded, batch_lengths_tensor)
                
                # Get predicted labels
                batch_predicted = batch_pred.argmax(dim=1).cpu()
                all_predictions.append(batch_predicted)
            
            # Concatenate all batch predictions
            predicted_labels = torch.cat(all_predictions)
            
            return predicted_labels
    
    def run(self):
        """Run the model training and testing process"""
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'plotting_data': self.plotting_data}