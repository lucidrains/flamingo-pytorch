import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        num_latents = 64
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        einstein notation
        b - batch
        m - number of medias
        n - sequence
        d - dimension
        """
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(self.latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        lk, lv = self.to_kv(self.latents).chunk(2, dim = -1)

        k, v = self.to_kv(x).chunk(2, dim = -1)

        q = rearrange(q, 'n (h d) -> h n d', h = h)
        k, v = rearrange_many((k, v), 'b m n (h d) -> b h m n d', h = h)
        lk, lv = repeat_many((lk, lv), 'n (h d) -> b h m n d', b = b, m = m, h = h)

        k = torch.cat((k, lk), dim = -2)
        v = torch.cat((v, lv), dim = -2)

        q = q * self.scale

        # attention

        sim = einsum('h i d, b h m j d  -> b h m i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h m n d -> b m n (h d)', h = h)
        return self.to_out(out)
