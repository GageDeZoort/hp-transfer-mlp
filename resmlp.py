from __future__ import annotations

from math import sqrt

from torch.nn import Linear, Module, ModuleList, ReLU, Tanh, init


class ResMLP(Module):
    """Fully connected NN w/ residual connections and Gaussian init
    Args:
    in_dim: input dimension
    out_dim: output dimension
    width: # neurons per internal layer
    beta: strength of the residual connection
    gamma_0: tuning of final layer output normalisation
    depth: number of hidden layers
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int,
        beta: float = 1.0,
        gamma_0: float = 1.0,
        eta_0: float = 0.01,
        depth: int = 4,
        activation: Module = Tanh,
        optimizer: str = "adam",
        **kwargs,
    ):
        super(ResMLP, self).__init__()

        self.layers = ModuleList()
        for layer in range(depth + 1):
            self.layers.append(
                Linear(
                    in_dim if (layer == 0) else width,
                    out_dim if (layer == depth) else width,
                    bias=False,
                )
            )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        self.beta = beta
        self.gamma_0 = gamma_0
        self.eta_0 = eta_0
        self.gamma = gamma_0 * sqrt(width)
        self.depth = depth
        self.act = activation()

        self.lr = self.get_lr(optimizer)
        
        self.reset_parameters()

    def reset_parameters(self):
        for _, weights in enumerate(self.layers):
            for p in weights.weight:
                init.normal_(p.data, mean=0, std=1)

    def get_lr(self, optimizer):
        if "sgd" in optimizer.lower():
            return self.eta_0 * self.gamma_0**2 * self.width
        elif "adam" in optimizer.lower():
            return self.eta_0 * self.gamma_0 * sqrt(self.width)
        else:
            raise Exception(f"Cannot locate parametrization for optimizer {optimizer}")
        
    def forward(self, x: Tensor):
        for layer, weights in enumerate(self.layers):
            if layer == 0:
                x = weights(x) / sqrt(self.in_dim)
            elif (layer > 0) and (layer < self.depth):
                x = x + (self.beta / sqrt(self.depth * self.width)) * weights(
                    self.act(x)
                )
            else:
                x = weights(self.act(x)) / (self.width * self.gamma)
        return x
