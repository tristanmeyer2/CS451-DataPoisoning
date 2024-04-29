import torch
from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):

    def __init__(self, n_features, n_hidden):
        super().__init__()
        
        self.pipeline = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.pipeline(X)