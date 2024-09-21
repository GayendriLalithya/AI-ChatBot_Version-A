import torch
import torch.nn as nn
import pickle


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def load_model(model_path, input_dim, output_dim):
    hidden_dim = 128
    model = IntentClassifier(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    return model


def load_label_encoder(path):
    with open(path, 'rb') as f:
        label_encoder = pickle.load(f)
    return label_encoder


def load_vectorizer(vectorizer_path, tokenizer_func):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        # Assuming the vectorizer expects a tokenizer function to be set after loading
        vectorizer.tokenizer = tokenizer_func
        return vectorizer