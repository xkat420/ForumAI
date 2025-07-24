import json
import pickle
import re
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from typing import Dict, List, Callable, Any, Optional, Tuple
from collections import Counter

# Custom vocabulary for tokenization without relying on external models
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'
        
        # Initialize special tokens
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)
        self.add_token(self.eos_token)
    
    def add_token(self, token):
        if token not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            return idx
        return self.word2idx[token]
    
    def add_document(self, document):
        # Tokenize document and update vocabulary
        tokens = self.tokenize(document)
        for token in tokens:
            self.word_counts[token] += 1
    
    def tokenize(self, text):
        # Simple tokenization by splitting on whitespace and punctuation
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Replace punctuation with space
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Split into tokens
        return text.split()
    
    def build(self, min_freq=2, max_vocab_size=10000):
        # Build vocabulary from collected counts
        # Filter by frequency and limit size
        tokens = [token for token, count in self.word_counts.most_common(max_vocab_size) 
                 if count >= min_freq]
        
        # Reset vocabulary keeping only special tokens
        self.word2idx = {}
        self.idx2word = {}
        
        # Re-add special tokens
        self.add_token(self.pad_token)
        self.add_token(self.unk_token)
        self.add_token(self.eos_token)
        
        # Add filtered tokens
        for token in tokens:
            self.add_token(token)
    
    def encode(self, text, max_length=None):
        tokens = self.tokenize(text)
        ids = [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
        
        if max_length is not None:
            if len(ids) >= max_length:
                ids = ids[:max_length-1] + [self.word2idx[self.eos_token]]
            else:
                ids = ids + [self.word2idx[self.eos_token]]
                ids = ids + [self.word2idx[self.pad_token]] * (max_length - len(ids))
        
        return ids
    
    def decode(self, ids):
        tokens = [self.idx2word.get(idx, self.unk_token) for idx in ids]
        # Remove padding and stop at EOS token
        if self.eos_token in tokens:
            tokens = tokens[:tokens.index(self.eos_token)]
        tokens = [t for t in tokens if t != self.pad_token]
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.word2idx)

# Dataset for forum posts
class ForumDataset(Dataset):
    def __init__(self, texts, vocab, seq_length=64):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Encode text
        ids = self.vocab.encode(text, self.seq_length)
        
        # Create input and target sequences for language modeling
        # Input: all tokens except last, Target: all tokens except first
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        
        return x, y

# Simple RNN-based language model
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_length, embedding_dim)
        output, (hidden, cell) = self.rnn(embedded)  # output: (batch_size, seq_length, hidden_dim)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return logits
    
    def generate(self, vocab, seed_text='', max_length=100, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Tokenize and encode seed text
            if seed_text:
                input_ids = vocab.encode(seed_text)
                input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            else:
                # Start with just the EOS token if no seed text
                input_tensor = torch.tensor([[vocab.word2idx[vocab.eos_token]]], dtype=torch.long)
            
            generated_ids = input_ids.copy() if seed_text else []
            
            # Generate text token by token
            for _ in range(max_length):
                embedded = self.embedding(input_tensor)
                output, (hidden, cell) = self.rnn(embedded)
                logits = self.fc(output[:, -1, :])  # Get predictions for last token
                
                # Apply temperature scaling
                scaled_logits = logits / temperature
                
                # Sample from the distribution
                probs = F.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if EOS token is generated
                if next_token == vocab.word2idx[vocab.eos_token]:
                    break
                
                generated_ids.append(next_token)
                
                # Update input tensor for next iteration
                input_tensor = torch.tensor([[next_token]], dtype=torch.long)
            
            # Decode generated tokens
            return vocab.decode(generated_ids)

class TextModel:
    """RNN-based text generation model."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """Initialize the text model with optional progress callback."""
        self.vocab = Vocabulary()
        self.progress_callback = progress_callback
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def _cleanup_memory(self):
        """Clean up memory to prevent OOM errors."""
        if hasattr(torch.cuda, 'empty_cache') and self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def _report_progress(self, progress_type: str, data: Dict[str, Any]):
        """Report progress via callback if provided."""
        if self.progress_callback:
            self.progress_callback(progress_type, data)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for training or generation."""
        # Filter out the specific comment
        text = re.sub(r'\[attachment deleted by admin, limit reached\]', '', text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def _load_data(self, data_file: str) -> List[str]:
        """Load and preprocess data from JSON file."""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        for post in data.get("posts", []):
            content = post.get("content", "")
            if content and len(content) > 6000:  # Filter very short posts
                texts.append(self._preprocess_text(content))
        
        return texts
    
    def train(self, data_file: str, epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the model on forum data."""
        # Load and preprocess data
        texts = self._load_data(data_file)
        
        if not texts:
            raise ValueError("No valid texts found in the data file.")
        
        # Build vocabulary
        print("Building vocabulary...")
        for text in texts:
            self.vocab.add_document(text)
        
        self.vocab.build(min_freq=2)  # Only keep words that appear at least twice
        vocab_size = len(self.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Create dataset and dataloader
        dataset = ForumDataset(texts, self.vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = RNNLanguageModel(vocab_size).to(self.device)
        
        # Setup optimizer
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.word2idx[self.vocab.pad_token])
        
        # Report start of training
        self._report_progress("start", {
            "total_steps": len(dataloader) * epochs,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_texts": len(texts),
            "vocab_size": vocab_size
        })
        
        # Clean up memory before training
        self._cleanup_memory()
        
        # Training loop
        self.model.train()
        global_step = 0
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                try:
                    # Move batch to device
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)  # (batch_size, seq_length, vocab_size)
                    
                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, vocab_size)
                    targets = targets.reshape(-1)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    
                    # Update loss
                    loss_value = loss.item()
                    total_loss += loss_value
                    epoch_loss += loss_value
                    
                    # Report progress
                    global_step += 1
                    if global_step % 10 == 0:
                        self._report_progress("step", {
                            "step": global_step,
                            "loss": loss_value,
                            "epoch": epoch
                        })
                        
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"Memory error encountered. Skipping batch and reducing memory usage.")
                        # Free memory
                        self._cleanup_memory()
                        # Zero gradients to recover
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
            
            # Report epoch completion
            avg_epoch_loss = epoch_loss / len(dataloader)
            self._report_progress("epoch", {
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "step": global_step
            })
            
            # Generate sample text after each epoch
            sample_text = self.generate(max_length=50, temperature=0.7)
            print(f"Sample text after epoch {epoch+1}: {sample_text}")
        
        # Report completion
        self._report_progress("complete", {
            "total_steps": len(dataloader) * epochs,
            "final_loss": total_loss / max(1, global_step)
        })
        
        # Clean up memory after training
        self._cleanup_memory()
    
    def generate(self, prompt: str = "", max_length: int = 100, temperature: float = 0.7) -> str:
        """Generate text based on optional prompt with memory optimization."""
        if not self.model or not self.vocab:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        try:
            # Clean up memory before generation
            self._cleanup_memory()
            
            # Preprocess prompt
            if prompt:
                prompt = self._preprocess_text(prompt)
                # Generate text using the RNN model's generate method
                generated_text = self.model.generate(self.vocab, seed_text=prompt, max_length=max_length, temperature=temperature)
            else:
                # Generate text without a prompt
                generated_text = self.model.generate(self.vocab, max_length=max_length, temperature=temperature)
            
            # Clean up memory after generation
            self._cleanup_memory()
            
            return generated_text
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e):
                print(f"Memory error during text generation: {str(e)}")
                self._cleanup_memory()
                # Try with CPU if CUDA out of memory
                if self.device.type == 'cuda':
                    print("Attempting generation on CPU instead...")
                    original_device = self.device
                    self.device = torch.device("cpu")
                    self.model.to(self.device)
                    
                    try:
                        if prompt:
                            result = self.model.generate(self.vocab, seed_text=prompt, max_length=min(max_length, 50), temperature=temperature)
                        else:
                            result = self.model.generate(self.vocab, max_length=min(max_length, 50), temperature=temperature)
                        
                        # Move model back to original device
                        self.device = original_device
                        self.model.to(self.device)
                        return result
                    except Exception:
                        # Move model back to original device
                        self.device = original_device
                        self.model.to(self.device)
                
                return f"Error: Not enough memory to generate text. Try a shorter prompt or reduce max_length."
            else:
                raise e
    
    def save_model(self, filename: str):
        """Save model and vocabulary to file."""
        if not self.model or not self.vocab:
            raise ValueError("No model to save. Please train a model first.")
        
        # Save model state and vocabulary
        model_data = {
            "model_state": self.model.state_dict(),
            "vocab": self.vocab,
            "embedding_dim": self.model.embedding.embedding_dim,
            "hidden_dim": self.model.rnn.hidden_size,
            "num_layers": self.model.rnn.num_layers,
            "dropout": self.model.dropout.p
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        return filename
    
    def load_model(self, filename: str):
        """Load model and vocabulary from file."""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vocab = model_data["vocab"]
            
            # Get model parameters
            vocab_size = len(self.vocab)
            embedding_dim = model_data.get("embedding_dim", 128)
            hidden_dim = model_data.get("hidden_dim", 256)
            num_layers = model_data.get("num_layers", 2)
            dropout = model_data.get("dropout", 0.2)
            
            # Initialize model with the same configuration as the saved model
            self.model = RNNLanguageModel(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(self.device)
            
            # Load state dict
            try:
                self.model.load_state_dict(model_data["model_state"])
                print(f"Model loaded successfully from {filename}")
            except Exception as e:
                print(f"Error loading model state: {str(e)}")
                print("Using newly initialized model with saved vocabulary")
            
            return self
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Initializing a new model and vocabulary from scratch...")
            
            # Initialize a new vocabulary and model
            self.vocab = Vocabulary()
            self.model = None
            
            return self

# Example usage
if __name__ == "__main__":
    model = TextModel()
    # Train model
    # model.train('forum_data.json')
    # model.save_model('forum_model.pkl')
    
    # Load model and generate text
    # model.load_model('forum_model.pkl')
    # text = model.generate("Hello", max_length=100, temperature=0.7)
    # print(text)