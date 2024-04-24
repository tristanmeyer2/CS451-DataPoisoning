import torch

class PrepareData():
    X, y = None, None
    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.tensor(X.values, dtype= torch.float)
        if not torch.is_tensor(y):
            self.y = torch.tensor(y.values, dtype= torch.float).reshape(len(y), 1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    