import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """A lightweight style statistics mixing module."""

    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if torch.rand(1, device=x.device).item() > self.p:
            return x
        if x.dim() != 4:
            return x

        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = torch.sqrt(var + self.eps)

        x_norm = (x - mu) / sig

        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        lam = self.beta.sample((b, 1, 1, 1)).to(x.device)
        mu_mix = mu * lam + mu2 * (1.0 - lam)
        sig_mix = sig * lam + sig2 * (1.0 - lam)
        return x_norm * sig_mix + mu_mix

