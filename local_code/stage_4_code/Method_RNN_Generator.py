'''
Concrete MethodModule class for joke text generation using RNN
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import time
from collections import Counter
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

possible_jokes = [
    "i walked into a bar",
    "my wife said i was",
    "the teacher asked me why",
    "why don't scientists trust atoms",
    "did you hear about the",
    "the waiter asked me if",
    "my dog looked at me",
    "when life gives you lemons",
    "the police officer pulled me",
    "my therapist says i have",
    "two penguins walk into a",
    "my grandmother always told me",
    "the alien looked confused when",
    "my boss caught me sleeping",
    "the barista winked at me"
]

class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-3
    word_to_idx = {}
    idx_to_word = {}
    vocab_size = 0
    max_seq_length = 50  # Maximum sequence length for a joke
    embedding_dim = 128  # Larger embedding for better text generation
    hidden_dim = 256     # Larger hidden dimension for generation
    num_layers = 2
    teacher_forcing_ratio = 0.5  # Use teacher forcing 50% of the time
    temperature = 0.8  # Controls randomness in sampling (lower = more deterministic)

    def __init__(self, mName, mDescription, max_epoch=100, learning_rate=1e-3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        # Store parameters
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        
        # Initialize other attributes
        self.plotting_data = {
            'epoch': [],
            'train_loss': [],
        }
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA device count: {torch.cuda.device_count()}')
            print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
        self.to(self.device)
        print(f'Using device: {self.device}')

    def _build_vocab(self, jokes, min_freq=1):
        """Build vocabulary from all jokes"""
        print('Building vocabulary for generator...')
        word_counts = Counter()
        for joke in jokes:
            words = joke.lower().split()
            word_counts.update(words)

        # For generation, we want to keep more words, so lower min_freq
        filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Special tokens for generation
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        for idx, word in enumerate(filtered_words, start=len(self.word_to_idx)):
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
            if len(words) > self.max_seq_length - 2:  # -2 for SOS and EOS tokens
                words = words[:self.max_seq_length-2]
            
            # Add SOS and EOS tokens
            words = ['<SOS>'] + words + ['<EOS>']
            
            # Convert to indices
            indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
            sequences.append(indices)
            sequence_lengths.append(len(indices))
            
        return sequences, sequence_lengths

    def _build_model(self):
        """Initialize model layers after vocabulary is built"""
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        
        # RNN layer (using LSTM for better sequence modeling)
        self.rnn = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.3 if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer maps hidden states to vocabulary
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
        
        # Move model to device
        self.to(self.device)

    def forward(self, input_seq, hidden=None):
        """Forward pass for a single time step"""
        # input_seq shape: [batch_size, 1]
        batch_size = input_seq.size(0)
        
        # Embed the input
        embedded = self.embedding(input_seq)  # [batch_size, 1, embedding_dim]
        
        # Get RNN outputs and new hidden state
        rnn_output, hidden = self.rnn(embedded, hidden)
        
        # Apply output layer
        output = self.fc(rnn_output.squeeze(1))  # [batch_size, vocab_size]
        
        return output, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))

    def train_model(self, jokes):
        """Train the generator model on jokes"""
        # First build vocabulary if not done yet
        if self.vocab_size == 0:
            self._build_vocab(jokes)
            self._build_model()
        
        # Convert jokes to sequences of indices
        joke_sequences, joke_lengths = self._text_to_indices(jokes)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding in loss calculation
        
        # Define batch size
        batch_size = 32  # Smaller batch size for more frequent updates
        
        # Training loop
        for epoch in range(self.max_epoch):
            start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            # Set model to training mode
            self.train()
            
            # Create batches
            indices = np.random.permutation(len(joke_sequences))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_size_actual = len(batch_indices)
                
                # Sort by length for pack_padded_sequence
                batch_indices = sorted(batch_indices, key=lambda x: joke_lengths[x], reverse=True)
                
                # Get batch data
                batch_jokes = [joke_sequences[j] for j in batch_indices]
                batch_lengths = [joke_lengths[j] for j in batch_indices]
                
                # Prepare input and target sequences
                # Input: all tokens except the last one (EOS)
                # Target: all tokens except the first one (SOS)
                input_seqs = [seq[:-1] for seq in batch_jokes]
                target_seqs = [seq[1:] for seq in batch_jokes]
                
                # Pad sequences
                input_padded = pad_sequence([torch.LongTensor(seq) for seq in input_seqs], 
                                           batch_first=True, padding_value=0)
                target_padded = pad_sequence([torch.LongTensor(seq) for seq in target_seqs], 
                                            batch_first=True, padding_value=0)
                
                # Move to device
                input_padded = input_padded.to(self.device)
                target_padded = target_padded.to(self.device)
                
                # Initialize hidden state
                hidden = self.init_hidden(batch_size_actual)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = []
                for t in range(input_padded.size(1)):
                    input_t = input_padded[:, t].unsqueeze(1)  # [batch_size, 1]
                    output, hidden = self.forward(input_t, hidden)
                    outputs.append(output)
                
                # Stack outputs along the sequence dimension
                outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, self.vocab_size)
                target_padded = target_padded.reshape(-1)
                
                # Calculate loss
                loss = criterion(outputs, target_padded)
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # Store metrics for plotting
            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['train_loss'].append(avg_loss)
            
            # Print progress
            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000  # convert to milliseconds
            print(f'Epoch: {epoch}/{self.max_epoch} | Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}ms')
            
            sample_output = self.generate_joke(random.choice(possible_jokes))
            print(f"Sample joke: {sample_output}")

    def generate_joke(self, start_text, max_length=50):
        """Generate a joke given the first few words"""
        if self.vocab_size == 0:
            return "Model has not been trained yet."
        
        # Set model to evaluation mode
        self.eval()
        
        # Convert start text to lowercase and tokenize
        words = start_text.lower().split()
        
        # Convert words to indices
        current_indices = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Initialize hidden state
        hidden = self.init_hidden(1)
        
        with torch.no_grad():
            # Process the input sequence
            for i in range(len(current_indices)):
                input_tensor = torch.LongTensor([[current_indices[i]]]).to(self.device)
                _, hidden = self.forward(input_tensor, hidden)
            
            # Start generating new words
            eos_id = self.word_to_idx['<EOS>']
            for _ in range(max_length - len(words)):
                # Get the last generated word
                input_tensor = torch.LongTensor([[current_indices[-1]]]).to(self.device)
                
                # Forward pass
                output, hidden = self.forward(input_tensor, hidden)
                
                # Apply temperature to logits
                scaled_output = output / self.temperature
                
                # Sample from the distribution
                probs = torch.softmax(scaled_output, dim=1)
                top_k = 5  # Consider only top k words for diversity
                top_indices = torch.topk(probs, top_k).indices[0]
                top_probs = probs[0, top_indices] / torch.sum(probs[0, top_indices])
                next_token = top_indices[torch.multinomial(top_probs, 1).item()]
                
                # Stop if EOS token is generated
                if next_token.item() == eos_id:
                    break
                
                # Add the generated word to the sequence
                current_indices.append(next_token.item())
        
        # Convert indices back to words
        generated_words = [self.idx_to_word.get(idx, '<UNK>') for idx in current_indices]
        
        # Filter out special tokens and join to form the joke
        generated_text = ' '.join([word for word in generated_words 
                                 if word not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
        
        return generated_text

    def run(self):
        """Run the model training process"""
        print('Generator method running...')
        print('--start training...')
        
        # Train on all jokes
        self.train_model(self.data['jokes'])
        
        # Generate jokes from the sample starts
        generated_jokes = {}
        for start in possible_jokes:
            joke = self.generate_joke(start)
            generated_jokes[start] = joke
            print(f"Input: '{start}'\nGenerated: '{joke}'\n")
        
        # Add some random starts that might not be in the dataset
        random_starts = [
            "three cats were sitting on the",
            "the doctor said you need to",
            "my computer keeps showing an error",
            "never trust a person who tells",
            "if I could travel back in"
        ]
        
        # Generate jokes from random starts
        for start in random_starts:
            joke = self.generate_joke(start)
            generated_jokes[start] = joke
            print(f"Random Input: '{start}'\nGenerated: '{joke}'\n")
        
        return {
            'generated_jokes': generated_jokes,
            'plotting_data': self.plotting_data
        }
