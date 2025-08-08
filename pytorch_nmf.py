import torch

# Custom PyTorch NMF Implementation (GPU-optimized)
class PyTorchNMF:
    def __init__(self, W, H, shape, rank, max_iter=200):
        self.W = W
        self.H = H
        self.shape = shape
        self.rank = rank
        self.max_iter = max_iter
        self.device = torch.device("mps")

    def fit(self, V):
        V = V.to(self.device)
        for _ in range(self.max_iter):
            # Update H
            V_approx = torch.matmul(self.W, self.H)
            self.H *= torch.matmul(self.W.T, V / (V_approx + 1e-10)) / torch.sum(self.W.T, dim=1, keepdim=True)
            self.H = torch.clamp(self.H, min=0)
            # Update W
            V_approx = torch.matmul(self.W, self.H)
            self.W *= torch.matmul(V / (V_approx + 1e-10), self.H.T) / torch.sum(self.H.T, dim=0, keepdim=True)
            self.W = torch.clamp(self.W, min=0)
        return self

    def __call__(self):
        return torch.matmul(self.W, self.H)


