from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np
import time
import random
from transformers import BertTokenizer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create matrix of shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Create position matrix of shape [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create division term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # Make sure dimensions match
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        return x + self.pe[:, :x.size(1), :].expand(x.size(0), -1, -1)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = x.transpose(0, 1)  # MultiheadAttention expects seq_len first
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)  # Restore batch first
        return self.layer_norm(x.transpose(0, 1) + attn_output)  # Add residual connection

class Method_RNN_Generator(method, nn.Module):
    data = None
    max_epoch = 20  # Increased from 10
    learning_rate = 1e-3
    
    def __init__(self, mName, mDescription, vocab_size=None, 
                 embedding_dim=512, hidden_dim=512, num_layers=6, 
                 max_epoch=10, learning_rate=1e-3):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        print("Loading BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Add special tokens that BERT might not have
        special_tokens = {
            'bos_token': '[BOS]',
            'eos_token': '[EOS]',
            'pad_token': '[PAD]',
            'sep_token': '[SEP]',
            'cls_token': '[CLS]'
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        self.vocab_size = self.tokenizer.vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        
        # Enhanced embedding with positional encoding
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(embedding_dim, self.vocab_size)  # Use embedding_dim
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)  # Use embedding_dim
        self.layer_norm2 = nn.LayerNorm(embedding_dim)  # Use embedding_dim
        
        # Dropout
        self.dropout = nn.Dropout(p=0.1)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f'Using device: {self.device}')
        
        self.plotting_data = {
            'epoch': [],
            'loss': [],
            'perplexity': [],
            'training_time': 0
        }
        
        self.init_weights()

    def init_weights(self):
        """Enhanced weight initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param)  # Better for RNNs
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif param.dim() >= 2:
                nn.init.kaiming_normal_(param)  # Better for deep networks
            else:
                nn.init.uniform_(param, -0.1, 0.1)

    def forward(self, x):
        # Get batch size and sequence length
        batch_size, seq_length = x.shape
        
        # Create padding mask (1 for padding tokens, 0 for real tokens)
        padding_mask = (x == self.tokenizer.pad_token_id)
        
        # Embedding
        embedded = self.embedding(x)  # Shape: [batch_size, seq_length, embedding_dim]
        embedded = self.positional_encoding(embedded)
        
        # Apply transformer layers with padding mask
        output = self.transformer(
            embedded,
            src_key_padding_mask=padding_mask
        )
        
        # Final layer norm and output projection
        output = self.layer_norm2(output)
        output = self.fc(output)
        
        return output, None

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
    
    def generate_text(self, seed_text, max_length=100, temperature=0.9, top_p=0.95):
        self.set_evaluation_mode()
        
        # Add beginning of sequence token to seed
        seed_text = f"{self.tokenizer.cls_token} {seed_text}"
        input_ids = self.tokenize_text(seed_text)
        current_ids = input_ids.unsqueeze(0).to(self.device)
        
        generated_tokens = []
        min_tokens = 10  # Ensure we generate at least this many tokens
        
        with torch.no_grad():
            for i in range(max_length):
                output, _ = self.forward(current_ids)
                last_token_logits = output[0, -1, :]
                
                # Filter out special tokens and unwanted characters
                for token_id in range(self.vocab_size):
                    token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                    # Filter special tokens
                    if token_id in [self.tokenizer.pad_token_id, self.tokenizer.unk_token_id,
                                  self.tokenizer.bos_token_id, self.tokenizer.cls_token_id,
                                  self.tokenizer.sep_token_id, self.tokenizer.mask_token_id]:
                        last_token_logits[token_id] = float('-inf')
                    # Only filter obvious special characters
                    elif any(c in token for c in ['[', ']', '*', '#', '@', '/']):
                        last_token_logits[token_id] = float('-inf')
                
                # Temperature and nucleus sampling
                scaled_logits = last_token_logits / temperature
                probs = torch.softmax(scaled_logits, dim=-1)
                
                # Nucleus (top-p) sampling
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()
                
                next_token_id = torch.multinomial(probs, 1)
                token = self.tokenizer.convert_ids_to_tokens([next_token_id.item()])[0]
                
                # Only stop for period if we've generated minimum tokens
                if len(generated_tokens) >= min_tokens:
                    if (next_token_id.item() == self.tokenizer.eos_token_id or 
                        (token == '.' and len(generated_tokens) > min_tokens)):
                        break
                
                generated_tokens.append(next_token_id.item())
                current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0)], dim=1)
                
                # Keep reasonable sequence length
                if current_ids.size(1) > 128:
                    current_ids = current_ids[:, -128:]
        
        # Decode and clean up
        generated_text = self.tokenizer.decode(generated_tokens, 
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True)
        
        # Clean up the text
        generated_text = generated_text.replace(self.tokenizer.cls_token, "").replace(self.tokenizer.sep_token, "")
        generated_text = generated_text.replace('[', '').replace(']', '').replace('  ', ' ')
        generated_text = ' '.join(word for word in generated_text.split() 
                                if not any(c in word for c in ['[', ']', '#', '@', '*']))
        
        # Combine with seed text (removing special tokens)
        seed_without_special = seed_text.replace(self.tokenizer.cls_token, "").strip()
        full_text = seed_without_special + " " + generated_text
        
        # Add period if missing
        if not full_text.strip().endswith(('.', '!', '?')):
            full_text += '.'
        
        return full_text.strip()
    
    def calculate_accuracy(self, output, target):
        """Calculate word prediction accuracy"""
        predictions = output.argmax(dim=1)
        valid_tokens = (target != self.tokenizer.pad_token_id)  # Ignore padding
        correct = (predictions == target) & valid_tokens
        accuracy = correct.sum().float() / valid_tokens.sum()
        return accuracy.item()

    def train_model(self, X, y):
        print("Initializing training...")
        training_start_time = time.time()
        
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Enhanced optimizer with higher weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.1,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=2,  # Restart every 2 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6
        )
        
        loss_function = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        batch_size = 32
        gradient_accumulation_steps = 2
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        num_samples = X.size(0)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        print(f"Training with batch size: {batch_size} (effective batch size: {effective_batch_size})")
        print(f"Number of batches: {num_batches}")
        
        for epoch in range(self.max_epoch):
            epoch_start_time = time.time()
            self.set_training_mode()
            epoch_loss = 0
            optimizer.zero_grad()
            
            print(f"\nEpoch {epoch+1}/{self.max_epoch}")
            
            # Shuffle data
            indices = torch.randperm(num_samples)
            X = X[indices]
            y = y[indices]
            
            for batch in range(num_batches):
                batch_start_time = time.time()
                
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, num_samples)
                
                batch_X = X[start_idx:end_idx].to(self.device)
                batch_y = y[start_idx:end_idx].to(self.device)
                
                # Use mixed precision training if CUDA is available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output, _ = self.forward(batch_X)
                        output = output.view(-1, self.vocab_size)
                        target = batch_y.contiguous().view(-1)
                        loss = loss_function(output, target)
                        loss = loss / gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    if (batch + 1) % gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    # Regular training for CPU
                    output, _ = self.forward(batch_X)
                    output = output.view(-1, self.vocab_size)
                    target = batch_y.contiguous().view(-1)
                    loss = loss_function(output, target)
                    loss = loss / gradient_accumulation_steps
                    
                    loss.backward()
                    
                    if (batch + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                if batch % 10 == 0:
                    batch_time = time.time() - batch_start_time
                    perplexity = torch.exp(loss)
                    print(f"  Batch {batch+1}/{num_batches}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Perplexity: {perplexity:.2f}, "
                          f"Time: {batch_time:.2f}s")
            
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / num_batches
            perplexity = torch.exp(torch.tensor(avg_loss))
            
            print(f"Epoch {epoch+1} complete - "
                  f"Time: {epoch_time:.2f}s, "
                  f"Loss: {avg_loss:.4f}, "
                  f"Perplexity: {perplexity:.2f}")
            
            # Generate sample text every 2 epochs
            if (epoch + 1) % 2 == 0:
                print("\nSample generations:")
                prompts = [
                    "Why did the chicken cross the road?",
                    "What do you call a bear with no teeth?",
                    "A man walks into a bar and says",
                    "Knock knock! Who's there?",
                ]
                for prompt in prompts:
                    sample = self.generate_text(prompt, max_length=50, temperature=0.7, top_p=0.9)
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {sample}")
            
            # Save metrics
            self.plotting_data['epoch'].append(epoch)
            self.plotting_data['loss'].append(avg_loss)

        total_time = time.time() - training_start_time
        self.plotting_data['training_time'] = total_time
        print(f"\nTraining finished in {total_time:.2f}s")

    def set_training_mode(self):
        """Enable training mode (dropout, batch norm, etc.)"""
        nn.Module.train(self, True)

    def set_evaluation_mode(self):
        """Enable evaluation mode (disable dropout, etc.)"""
        nn.Module.eval(self)

    def run(self):
        print('method running...')
        print('--start training...')
        
        # Get training data
        train_texts = self.data['train']['X']
        
        # Clean up the texts first
        train_texts = [text.strip() for text in train_texts]
        
        # Add special tokens in a way BERT understands
        train_texts = [f"{self.tokenizer.cls_token} {text} {self.tokenizer.sep_token}" 
                      for text in train_texts]
        
        print(f"Number of training texts: {len(train_texts)}")
        
        # Tokenize all texts
        print("Tokenizing texts...")
        tokenized_texts = [self.tokenize_text(text) for text in train_texts]
        
        # Find max length for padding
        max_length = max(len(tokens) for tokens in tokenized_texts)
        print(f"Max sequence length: {max_length}")
        
        # Reduce sequence length more aggressively
        max_length = min(max_length, 128)  # Reduced from 512 to 128
        
        print("Padding sequences...")
        padded_sequences = []
        for tokens in tokenized_texts:
            # Truncate if longer than max_length
            tokens = tokens[:max_length]
            # Pad if shorter
            if len(tokens) < max_length:
                padding = torch.zeros(max_length - len(tokens), dtype=torch.long)
                tokens = torch.cat([tokens, padding])
            padded_sequences.append(tokens)
        
        print("Creating tensors...")
        X = torch.stack(padded_sequences)
        y = X.clone()
        y[:, :-1] = X[:, 1:]
        y[:, -1] = self.tokenizer.pad_token_id
        
        print(f"Input shape: {X.shape}")
        print("Starting training...")
        self.train_model(X, y)
        
        print('--start generation...')
        generated_text = self.generate_text(seed_text="The", max_length=50)
        return {
            'generated_text': generated_text,
            'plotting_data': self.plotting_data
        }

    def tokenize_text(self, text):
        """Tokenize text using BERT tokenizer"""
        # Add special tokens and return tensor
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return encoded.squeeze(0)  # Remove batch dimension

    def decode_tokens(self, tokens):
        """Decode tokens back to text"""
        return self.tokenizer.decode(tokens)

    def create_positional_encoding(self):
        max_seq_length = 1024  # Maximum sequence length to support
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-math.log(10000.0) / self.embedding_dim))
        pe = torch.zeros(max_seq_length, self.embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe