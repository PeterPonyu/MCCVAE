import torch
import geoopt
from torch.nn import functional as F
from torch.distributions import Normal


class Distribution():
    def __init__(self, mean, sigma) -> None:
        self.mean = mean  # (1, *, 3)
        self.sigma = torch.nan_to_num(sigma, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-8)

        self.latent_dim = 2
        self.base = Normal(
            torch.zeros([*self.sigma.shape[:-1], 2], device=self.mean.device),
            self.sigma,
            validate_args=False,
        )
        self.manifold = geoopt.manifolds.Lorentz()
        self.origin = self.manifold.origin(
            self.mean.size(),
            device=self.mean.device
        )

        self.kl_div = None

    def log_prob(self, z):
        eps = 1e-8

        u = self.manifold.logmap(self.mean, z)
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)

        v = self.manifold.transp(self.mean, self.origin, u)
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        v_spatial = v[..., 1:]
        log_prob_v = self.base.log_prob(v_spatial).sum(dim=-1)

        r = self.manifold.norm(u).clamp_min(eps)
        ratio = (torch.sinh(r) / r).clamp_min(eps)
        log_det = (self.latent_dim - 1) * torch.log(ratio)

        return log_prob_v - log_det

    def rsample(self, N):
        v = self.base.rsample([N])  # (N, *, 2)
        v = F.pad(v, (1, 0))  # (N, *, 3)

        u = self.manifold.transp0(self.mean, v)  # (N, *, 3)
        z = self.manifold.expmap(self.mean, u)

        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

