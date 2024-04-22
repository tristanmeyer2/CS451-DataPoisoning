import torch
from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):

    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        
        self.pipeline = nn.Sequential(
            nn.Linear(n_features, 1)
        )

    def score(self, X):

        return self.pipeline(X)

    
    def forward(self, X):
        return F.sigmoid(self.pipeline(X)) 