import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from flamingo_pytorch.flamingo_pytorch import GatedCrossAttentionBlock, PerceiverResampler

# helper functions

def exists(val):
    return val is not None

# for controlling freezing during training of flamingo

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# transformer


class FlamingoPaLM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        media_token_id=3,
        cross_attn_every=3,
        img_encoder=None,
        perceiver_num_latents=64,
        perceiver_depth=2,
        max_video_frames = None,
        only_attend_immediate_media=True
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.media_token_id = media_token_id # you need to reserve a special token id for media

        self.video_frame_pos_emb = nn.Parameter(torch.randn(max_video_frames, dim)) if exists(max_video_frames) else None

        self.img_encoder = img_encoder
        freeze_model_and_make_eval_(self.img_encoder)

        self.perceiver_resampler = PerceiverResampler(
            dim=dim,
            depth=perceiver_depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=perceiver_num_latents
        )

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads, only_attend_immediate_media=only_attend_immediate_media) if not (ind % cross_attn_every) else None
            ]))

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)
    
    def forward(
        self,
        text,
        *,
        images=None,
        videos=None,
        embeds=None
    ):
        batch, device = text.shape[0], text.device

        flamingo_mode = any([exists(t) for t in (images, videos, embeds)])

        # automatically take care of freezing or unfreezing depending on what is passed in

        if flamingo_mode:
            # in flamingo mode, freeze everything but perceiver and gated cross attention
            freeze_all_layers_(self)
            unfreeze_all_layers_(self.perceiver_resampler)
            [unfreeze_all_layers_(cross_attn) for _, cross_attn in self.layers if exists(cross_attn)]
        else:
            unfreeze_all_layers_(self)

        # derive the media token ids (as a boolean tensor), for calculating the masked cross attention

        if flamingo_mode:
            media_locations = text == self.media_token_id

        text_tokens = self.token_emb(text)

        assert not (exists(embeds) and (exists(images) or exists(video)))

        # encode videos or images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image embeddings

        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            images = rearrange(images, 'b t ... -> (b t) ...')

            with torch.no_grad():
                embeds = self.img_encoder(images)

            embeds = rearrange(embeds, '(b t) ... -> b t ...', b = batch)

        if exists(videos):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic video encoding'
            batch, media, num_times, *_ = videos.shape
            videos = rearrange(videos, '... c h w -> (...) c h w')

            with torch.no_grad():
                embeds = self.img_encoder(videos)

            embeds = rearrange(embeds, '(b m t) ... -> b m t ...', b = batch, m = media, t = num_times)

            video_time_pos_emb = repeat(self.video_frame_pos_emb[:num_times], 't d -> b m t n d', b = batch, m = media, n = embeds.shape[-2])
            embeds = embeds + video_time_pos_emb
            embeds = rearrange(embeds, 'b m t n d -> b m (t n) d')

        if exists(embeds):
            embeds = self.perceiver_resampler(embeds)


        # go through layers

        for attn_ff, flamingo_cross_attn in self.layers:
            text_tokens = attn_ff(text_tokens)

            # if image embeds exist and flamingo cross attention set for the layer
            # do the cross attention
            if exists(flamingo_cross_attn) and exists(embeds):
                text_tokens = flamingo_cross_attn(
                    text_tokens,
                    embeds,
                    media_locations = media_locations
                )

        return self.to_logits(text_tokens)
