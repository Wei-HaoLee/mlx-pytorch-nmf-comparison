import mlx.core as mx

# Custom MLX NMF Implementation (GPU-optimized on Apple Silicon)
class MLXNMF:
    def __init__(self, W, H, shape, rank, max_iter=200):
        self.shape = shape
        self.rank = rank
        self.max_iter = max_iter
        self.W = W        
        self.H = H 

    def fit(self, V):
        for _ in range(self.max_iter):
            # Update H
            V_approx = mx.matmul(self.W, self.H, stream=mx.gpu)
            H_update = self.H * (mx.matmul(self.W.T, V / (V_approx + 1e-10), stream=mx.gpu) / mx.sum(self.W.T, axis=1, keepdims=True))
            self.H = mx.maximum(H_update, 0)
            # Update W
            V_approx = mx.matmul(self.W, self.H, stream=mx.gpu)
            W_update = self.W * (mx.matmul(V / (V_approx + 1e-10), self.H.T, stream=mx.gpu) / mx.sum(self.H.T, axis=0, keepdims=True))
            self.W = mx.maximum(W_update, 0)
            mx.eval(self.W, self.H)  # Force evaluation for MLX lazy computation
        return self

    def __call__(self):
        return mx.matmul(self.W, self.H, stream=mx.gpu)


