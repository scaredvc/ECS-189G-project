import torch
import torch.nn as nn
import numpy as np
import time
from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy # Assuming this will be used/adapted
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Method_RNN(method, nn.Module):
    data = None 
    
    def __init__(self, mName, mDescription,
                 vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers, bidirectional, dropout_prob, rnn_type='lstm', pad_idx=0,
                 max_epoch=10, learning_rate=1e-3, batch_size=64, device=None):
        
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim # For binary classification, output_dim=1 with BCEWithLogitsLoss is common
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.rnn_type = rnn_type.lower()
        assert self.rnn_type in ['lstm', 'gru'], "RNN type must be 'lstm' or 'gru'"
        self.pad_idx = pad_idx

        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        
        RnnCell = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU
        
        self.rnn = RnnCell(self.embedding_dim,
                             self.hidden_dim,
                             num_layers=self.n_layers,
                             bidirectional=self.bidirectional,
                             batch_first=True, # Expects (batch, seq, feature)
                             dropout=self.dropout_prob if self.n_layers > 1 else 0)
        
        fc_input_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.fc = nn.Linear(fc_input_dim, self.output_dim)
        
        self.dropout_layer = nn.Dropout(self.dropout_prob)

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        print(f'Method_RNN initialized for {self.method_name} on device: {self.device}')
        print(f'RNN Type: {self.rnn_type}, Bidirectional: {self.bidirectional}, Layers: {self.n_layers}, Hidden: {self.hidden_dim}')
        print(f'Embedding: {self.vocab_size}x{self.embedding_dim}, Output FC: {fc_input_dim}->{self.output_dim}')

        self.plotting_data = {'epoch': [], 'loss': [], 'accuracy': []}

    def forward(self, text_indices, text_lengths):
        # text_indices: (batch_size, seq_len)
        # text_lengths: (batch_size,) tensor of original sequence lengths
        
        embedded = self.dropout_layer(self.embedding(text_indices))
        # embedded: (batch_size, seq_len, embedding_dim)
        
        # Pack sequence - text_lengths should be on CPU
        packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'lstm':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
            # hidden and cell are (n_layers * num_directions, batch_size, hidden_dim)
        else: # gru
            packed_output, hidden = self.rnn(packed_embedded)
            # hidden is (n_layers * num_directions, batch_size, hidden_dim)

        # We use the final hidden state of the last layer
        # If bidirectional, hidden is (n_layers * 2, batch, hidden_dim).
        # The first n_layers are forward, the next n_layers are backward.
        # hidden[-1] is the last backward layer, hidden[-2] is the last forward layer if bidirectional.
        if self.bidirectional:
            # Concatenate the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden states of the last layer
            # LSTM: hidden state is tuple (h_n, c_n). We need h_n from the tuple.
            # GRU: hidden state is h_n directly.
            # The shape of hidden is (num_layers * num_directions, batch, hidden_size)
            # So hidden[-2] is the last layer, forward pass. hidden[-1] is the last layer, backward pass.
            last_hidden_state_forward = hidden[-2,:,:] 
            last_hidden_state_backward = hidden[-1,:,:]
            hidden = torch.cat((last_hidden_state_forward, last_hidden_state_backward), dim=1)
        else:
            # hidden is (num_layers, batch, hidden_size)
            hidden = hidden[-1,:,:] # Last layer, forward pass
            
        hidden = self.dropout_layer(hidden) # Apply dropout to the concatenated/final hidden state
        # hidden is now (batch_size, hidden_dim * num_directions)
        output = self.fc(hidden)
        # output: (batch_size, output_dim)
        return output

    def _train_one_epoch(self, train_loader, optimizer, loss_function, accuracy_evaluator):
        self.train() # Set model to training mode
        epoch_loss = 0
        epoch_preds_list = []
        epoch_true_list = []

        for batch_X, batch_y, batch_lengths in train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            # batch_lengths should remain on CPU for pack_padded_sequence
            
            optimizer.zero_grad()
            predictions_logits = self.forward(batch_X, batch_lengths) # (batch_size, output_dim)
            
            if self.output_dim == 1: # BCEWithLogitsLoss expects (N) and (N)
                loss = loss_function(predictions_logits.squeeze(), batch_y.float())
                predicted_labels = (torch.sigmoid(predictions_logits.squeeze()) > 0.5).long()
            else: # CrossEntropyLoss expects (N, C) and (N) for labels 0 to C-1
                loss = loss_function(predictions_logits, batch_y.long())
                predicted_labels = predictions_logits.argmax(dim=1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1) # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item() * batch_X.size(0)
            epoch_preds_list.extend(predicted_labels.cpu().numpy())
            epoch_true_list.extend(batch_y.cpu().numpy())
        
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        accuracy_evaluator.data = {'true_y': np.array(epoch_true_list), 'pred_y': np.array(epoch_preds_list)}
        epoch_accuracy = accuracy_evaluator.evaluate()
        return avg_epoch_loss, epoch_accuracy

    def train_model(self, X_indices_train, y_train, X_lengths_train, X_indices_val=None, y_val=None, X_lengths_val=None):
        if self.output_dim == 1:
            loss_function = nn.BCEWithLogitsLoss().to(self.device)
        else:
            loss_function = nn.CrossEntropyLoss().to(self.device)
            
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Assuming Evaluate_Accuracy class is available and works with numerical labels
        accuracy_evaluator = Evaluate_Accuracy('accuracy_evaluator', '') 

        train_dataset = torch.utils.data.TensorDataset(X_indices_train, y_train, X_lengths_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optional: Validation loader
        val_loader = None
        if X_indices_val is not None and y_val is not None and X_lengths_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_indices_val, y_val, X_lengths_val)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Starting training for {self.max_epoch} epochs...")
        for epoch in range(self.max_epoch):
            start_time = time.time()
            
            avg_train_loss, train_accuracy = self._train_one_epoch(train_loader, optimizer, loss_function, accuracy_evaluator)
            
            self.plotting_data['epoch'].append(epoch + 1)
            self.plotting_data['loss'].append(avg_train_loss)
            self.plotting_data['accuracy'].append(train_accuracy)
            
            elapsed_time = (time.time() - start_time) * 1000
            log_message = f'Epoch: {epoch+1}/{self.max_epoch}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Time: {elapsed_time:.0f}ms'

            if val_loader:
                val_loss, val_accuracy, _ = self._evaluate_model(val_loader, loss_function, accuracy_evaluator)
                log_message += f', Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}'
            
            print(log_message)
        print("Training complete.")

    def _evaluate_model(self, data_loader, loss_function, accuracy_evaluator):
        self.eval() # Set model to evaluation mode
        total_loss = 0
        all_preds_list = []
        all_true_list = []

        with torch.no_grad():
            for batch_X, batch_y, batch_lengths in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                # batch_lengths remain on CPU
                
                predictions_logits = self.forward(batch_X, batch_lengths)
                
                if self.output_dim == 1:
                    loss = loss_function(predictions_logits.squeeze(), batch_y.float())
                    predicted_labels = (torch.sigmoid(predictions_logits.squeeze()) > 0.5).long()
                else:
                    loss = loss_function(predictions_logits, batch_y.long())
                    predicted_labels = predictions_logits.argmax(dim=1)
                
                total_loss += loss.item() * batch_X.size(0)
                all_preds_list.extend(predicted_labels.cpu().numpy())
                all_true_list.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader.dataset)
        accuracy_evaluator.data = {'true_y': np.array(all_true_list), 'pred_y': np.array(all_preds_list)}
        accuracy = accuracy_evaluator.evaluate()
        return avg_loss, accuracy, np.array(all_preds_list)

    def test_model(self, X_indices_test, y_test, X_lengths_test):
        print("Evaluating on test set...")
        if self.output_dim == 1:
            loss_function = nn.BCEWithLogitsLoss().to(self.device)
        else:
            loss_function = nn.CrossEntropyLoss().to(self.device)
        accuracy_evaluator = Evaluate_Accuracy('accuracy_evaluator', '')

        test_dataset = torch.utils.data.TensorDataset(X_indices_test, y_test, X_lengths_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        avg_loss, accuracy, predictions = self._evaluate_model(test_loader, loss_function, accuracy_evaluator)
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return {'pred_y': predictions, 'true_y': y_test.cpu().numpy()} # Ensure y_test is also numpy array for consistency

    def run(self):
        # This method expects self.data to be pre-set by a Setting object.
        # self.data must contain preprocessed, numericalized, and padded data:
        # 'train': {'X_indices': torch.tensor, 'y': torch.tensor, 'lengths': torch.tensor}
        # 'test': {'X_indices': torch.tensor, 'y': torch.tensor, 'lengths': torch.tensor}
        # The Method_RNN instance should be initialized with appropriate vocab_size and output_dim.
        
        print(f'{self.method_name} ({self.method_description}) running for text classification...')
        
        if not self.data or 'train' not in self.data or 'test' not in self.data or \
           'X_indices' not in self.data['train'] or 'y' not in self.data['train'] or 'lengths' not in self.data['train'] or \
           'X_indices' not in self.data['test'] or 'y' not in self.data['test'] or 'lengths' not in self.data['test']:
            raise ValueError("Data not properly set or incomplete for Method_RNN.run(). Ensure preprocessed X_indices, y, and lengths are provided.")

        # Ensure data is on the correct device or can be moved.
        # For simplicity, we assume they are already tensors. If they are numpy arrays, convert them.
        def _to_tensor(data, dtype=torch.long):
            if isinstance(data, np.ndarray):
                return torch.tensor(data, dtype=dtype)
            return data # Assume already a tensor
        
        X_train_indices = _to_tensor(self.data['train']['X_indices'], dtype=torch.long) #.to(self.device) -- will be moved in train_loader loop
        y_train = _to_tensor(self.data['train']['y']) #.to(self.device)
        X_train_lengths = _to_tensor(self.data['train']['lengths'], dtype=torch.long) # Keep on CPU

        X_test_indices = _to_tensor(self.data['test']['X_indices'], dtype=torch.long) #.to(self.device)
        y_test = _to_tensor(self.data['test']['y']) #.to(self.device)
        X_test_lengths = _to_tensor(self.data['test']['lengths'], dtype=torch.long) # Keep on CPU

        print('--start training...')
        self.train_model(X_train_indices, y_train, X_train_lengths)
        
        print('--start testing...')
        test_results = self.test_model(X_test_indices, y_test, X_test_lengths)
        
        final_result = {
            'pred_y': test_results['pred_y'], 
            'true_y': test_results['true_y'],
            "plotting_data": self.plotting_data
        }
        print(f'{self.method_name} run complete.')
        return final_result

    # --- Placeholder for Text Generation --- 
    # def generate_story(self, starting_words_indices, word_to_idx, idx_to_word, max_length=100, temperature=1.0):
    #     self.eval()
    #     # ... implementation ...
    #     return "Generated story text"