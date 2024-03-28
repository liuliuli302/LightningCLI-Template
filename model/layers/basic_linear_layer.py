from torch import nn
from einops import rearrange


class BasicLinearModel(nn.Module):
    def __init__(self, in_features: int = 784, out_features: int = 10, hidden_dim: int = 512):
        super(BasicLinearModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=out_features)
        )

    def forward(self, x):
        x = rearrange(x, 'b 1 x y -> b (x y)', x=28, y=28)
        return self.model(x)
