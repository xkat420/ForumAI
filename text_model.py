import json
import pickle
import re
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from typing import Dict, List, Callable, Any, Optional, Tuple
from collections import Counter
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except AttributeError:
    nltk.download('punkt')

stemmer = LancasterStemmer()

class TextModel:
    """Chatbot model using a bag-of-words approach."""

    def __init__(self, progress_callback: Optional[Callable] = None):
        """Initialize the text model."""
        self.progress_callback = progress_callback
        self.model = None
        self.words = []
        self.labels = []
        self.training = []
        self.output = []
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

    def _load_data(self, data_file: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load data from JSON file."""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a structured format for intents
        intents = {"intents": []}
        for post in data.get("posts", []):
            if post.get("content"):
                intents["intents"].append({
                    "tag": f"post_{post['post_id']}",
                    "patterns": [post['title']] if post.get('title') else [],
                    "responses": [post['content']]
                })
        return intents

    def _prepare_data(self, data: Dict[str, List[Dict[str, Any]]]):
        """Tokenize, stem, and create bag-of-words."""
        docs_x = []
        docs_y = []

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                docs_x.append(tokens)
                docs_y.append(intent["tag"])

            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        self.words = [stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)

        out_empty = [0] * len(self.labels)

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(docs_y[x])] = 1

            self.training.append(bag)
            self.output.append(output_row)

        self.training = np.array(self.training)
        self.output = np.array(self.output)

    def train(self, data_file: str, epochs: int = 1000, batch_size: int = 8, learning_rate: float = 0.001):
        """Train the chatbot model."""
        data = self._load_data(data_file)
        self._prepare_data(data)

        input_size = len(self.training[0])
        output_size = len(self.output[0])

        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
            nn.Softmax(dim=1)
        ).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.training).float(), torch.from_numpy(self.output).float())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_steps = len(dataloader) * epochs
        self._report_progress("start", {"total_steps": total_steps})

        global_step = 0
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % 100 == 0:
                    self._report_progress("step", {"step": global_step, "loss": loss.item()})

        self._report_progress("complete", {"total_steps": total_steps, "final_loss": loss.item()})

    def _bag_of_words(self, s, words):
        bag = [0 for _ in range(len(words))]
        s_words = nltk.word_tokenize(s)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = 1
        return np.array(bag)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response to a prompt."""
        if not self.model:
            raise ValueError("Model not trained or loaded.")
            
        data = self._load_data('forum_data.json')

        input_data = torch.from_numpy(self._bag_of_words(prompt, self.words)).float().to(self.device)
        input_data = input_data.unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            results = self.model(input_data)
        
        results_index = torch.argmax(results)
        tag = self.labels[results_index]

        for intent in data["intents"]:
            if intent['tag'] == tag:
                return np.random.choice(intent['responses'])

    def save_model(self, filename: str):
        """Save the trained model."""
        data = {
            "model_state": self.model.state_dict(),
            "words": self.words,
            "labels": self.labels,
            "training": self.training,
            "output": self.output
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_model(self, filename: str):
        """Load a pre-trained model."""
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.words = data["words"]
        self.labels = data["labels"]
        self.training = data["training"]
        self.output = data["output"]

        input_size = len(self.training[0])
        output_size = len(self.output[0])

        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
            nn.Softmax(dim=1)
        ).to(self.device)
        self.model.load_state_dict(data["model_state"])
        self.model.eval()
