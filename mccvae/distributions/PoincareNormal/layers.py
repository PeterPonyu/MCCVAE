import torch
import geoopt
from torch import nn
from torch.nn import functional as F


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            3 * self.latent_dim
        )
        self.manifold = geoopt.manifolds.PoincareBall(1.0)

    def forward(self, feature):
        feature = self.variational(feature)
        mean, logsigma = torch.split(
            feature,
            [2 * self.latent_dim, self.latent_dim],
            dim=-1
        )

        mean = mean.view(*mean.shape[:-1], self.latent_dim, 2)
        mean = self.manifold.expmap0(mean)
        # Clamp logsigma to prevent numerical overflow in softplus
        # softplus(x) ≈ x for large x, so clamp at log(10) gives max scale ~10
        logsigma = torch.clamp(logsigma, min=-10.0, max=2.3)  # max scale ~10
        sigma = F.softplus(logsigma)[..., None]

        return mean, sigma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.manifold = geoopt.manifolds.PoincareBall(1.0)

    def forward(self, z):
        z = self.manifold.logmap0(z)
        z = z.reshape(*z.shape[:-2], -1)
        return z

