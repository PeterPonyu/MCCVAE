import torch
from math import log, sqrt, pi
from torch.distributions import Normal, Gamma
from torch.distributions.distribution import Distribution
from ..utils import euclidean_kl_div, gamma_kl_div

@torch.jit.script
def _log_prob(kl, log_gamma_square, log_beta_square, c):
    log_prob = -kl / log_gamma_square.exp() / (2 * -c)
    log_prob = log_prob - 1.5 * log_beta_square
    log_prob = log_prob - 0.5 * log_gamma_square

    gamma_factor = (-log_gamma_square).exp() / (4 * -c)
    log_prob = log_prob - gamma_factor
    log_prob = log_prob - torch.lgamma(gamma_factor)
    log_prob = log_prob - gamma_factor * ((4 * -c).log() + log_gamma_square)
    # log_prob = log_prob - 0.5 * log(2 * pi)

    return log_prob

class Distribution(Distribution):
    def __init__(self, means, log_gamma_square):
        super().__init__()
        # means: [..., 3] = (alpha, log_beta_square, log_c)
        self.alpha = means[..., 0]
        self.log_beta_square = means[..., 1]
        # c is curvature magnitude (>0), underlying manifold curvature is -c
        self.c = means[..., 2].exp()
        self.log_gamma_square = log_gamma_square

        # Normal base parameters
        self.normal_mu = self.alpha                    # mean
        self.normal_logvar = self.log_beta_square      # log variance

        # ---------- FIX: make Gamma parameters valid (both > 0) ----------
        denom = 4.0 * (self.c + 1e-6)                  # strictly positive
        self.gamma_a = (-self.log_gamma_square).exp() / denom + 1.0  # concentration > 0
        self.gamma_b = (-self.normal_logvar).exp() / denom           # rate > 0

        # ---------- FIX: define base1 so rsample() works ----------
        # Normal over alpha with variance exp(log_beta_square)
        self.base1 = Normal(
            loc=self.normal_mu,
            scale=(0.5 * self.normal_logvar).exp()     # std = exp(0.5 * logvar)
        )
        self.base2 = Gamma(self.gamma_a, self.gamma_b)

    def log_prob(self, z):
        target_mean, target_logvar = sqrt(-2 * self.c) * z[..., 0], z[..., 1]
        kl = euclidean_kl_div(
            target_mean, 
            target_logvar, 
            self.alpha * (-2 * self.c).sqrt(), 
            self.log_beta_square
        )
        return _log_prob(kl, self.log_gamma_square, self.log_beta_square, self.c)

    def rsample(self, N):
        sample_mean = self.base1.rsample([N])
        sample_logvar = self.base2.rsample([N]).log()
        return torch.stack([sample_mean, sample_logvar], dim=-1)

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

    def kl_div(self, target_dist):
        kl1 = euclidean_kl_div(
            self.normal_mu,
            self.normal_logvar,
            target_dist.normal_mu,
            target_dist.normal_logvar
        )
        kl2 = gamma_kl_div(
            self.gamma_a,
            self.gamma_b,
            target_dist.gamma_a,
            target_dist.gamma_b
        )
        return kl1 + kl2

