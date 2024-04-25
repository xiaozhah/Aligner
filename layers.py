import torch
from torch import nn

class LinearNorm(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        w_init_gain="linear",
        weight_norm=False,
        init_weight_norm=1.0,
    ):
        super(LinearNorm, self).__init__()
        if weight_norm:
            self.linear_layer = nn.utils.weight_norm(
                nn.Linear(in_dim, out_dim, bias=bias)
            )
            self.linear_layer.weight_g = nn.Parameter(torch.FloatTensor(1).fill_(init_weight_norm))
        else:
            self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)