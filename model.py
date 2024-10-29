from __future__ import annotations

from functools import partial
import math
import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

from einops import rearrange, einsum
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

# constants

from torch.nn.attention import SDPBackend

SDP_BACKEND_MAP = dict(
    enable_flash=SDPBackend.FLASH_ATTENTION,
    enable_mem_efficient=SDPBackend.EFFICIENT_ATTENTION,
    enable_math=SDPBackend.MATH,
    enable_cudnn=SDPBackend.CUDNN_ATTENTION,
)

def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def cast_tuple(t, length=1):
    out = t if isinstance(t, tuple) else ((t,) * length)
    assert len(out) == length
    return out


def l2norm(t, dim=-1, norm_eps=0.0, eps=None, groups=1):
    if groups > 1:
        t = t.chunk(groups, dim=dim)
        t = torch.stack(t)

    if norm_eps == 0.0:
        out = F.normalize(t, dim=dim, p=2)
    else:
        eps = default(eps, 1e-5 if t.dtype == torch.float16 else 1e-10)
        norm = t.norm(dim=dim, keepdim=True)
        target_norm = norm.detach().clamp(min=1.0 - norm_eps, max=1.0 + norm_eps)
        divisor = norm / target_norm
        out = t / divisor.clamp(min=eps)

    if groups > 1:
        out = torch.cat([*out], dim=dim)

    return out


class Scale(Module):
    """
    latter part of section 2.5 in the paper
    """

    def __init__(self, dim, init=1.0, scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale


class Residual(Module):
    def __init__(self, fn: Module, dim: int, init: float, scale: float):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim**-0.5))

    def forward(self, x, **kwargs):
        residual = x

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        out = l2norm(out)
        out = l2norm(residual.lerp(out, self.branch_scale()))

        if tuple_output:
            out = (out, *rest)

        return out


class L2Norm(Module):
    def __init__(self, dim=-1, norm_eps=0.0, groups=1):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps
        self.groups = groups

    def forward(self, t):
        return l2norm(t, dim=self.dim, norm_eps=self.norm_eps, groups=self.groups)


class NormLinear(Module):
    def __init__(
        self, dim, dim_out, norm_dim_in=True, parametrize=True, norm_eps=0.0, groups=1
    ):
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)

        self.scale = groups**-1
        self.parametrize = parametrize
        self.l2norm = L2Norm(
            dim=-1 if norm_dim_in else 0, norm_eps=norm_eps, groups=groups
        )

        if parametrize:
            register_parametrization(self.linear, "weight", self.l2norm)

        self.norm_weights_()

    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original

            original.copy_(normed)
        else:
            self.weight.copy_(self.l2norm(self.weight))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x) * self.scale


# attention


class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        norm_qk=True,
        causal=True,
        manual_norm_weights=False,
        s_qk_init=1.0,
        s_qk_scale=None,
        flash_kwargs: dict = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
        norm_eps=0.0,
        num_hyperspheres=1,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal

        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres,
        )
        self.l2norm = partial(l2norm, norm_eps=norm_eps, groups=num_hyperspheres)

        dim_sqrt = dim**0.5
        self.dim_sqrt = dim_sqrt
        self.attn_scale = dim_head**0.5

        dim_inner = dim_head * heads
        self.to_q = NormLinear_(dim, dim_inner)
        self.to_k = NormLinear_(dim, dim_inner)
        self.to_v = NormLinear_(dim, dim_inner)

        # flash attention related context manager

        sdpa_backends = [
            SDP_BACKEND_MAP[enable_str]
            for enable_str, enable in flash_kwargs.items()
            if enable
        ]
        self.sdpa_context_manager = partial(
            torch.nn.attention.sdpa_kernel, sdpa_backends
        )

        # qk rmsnorm + scale

        self.norm_qk = norm_qk
        self.q_scale = Scale(dim, s_qk_init, default(s_qk_scale, dim**-0.5))
        self.k_scale = Scale(dim, s_qk_init, default(s_qk_scale, dim**-0.5))

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        self.to_out = NormLinear_(dim_inner, dim, norm_dim_in=False)

    def forward(
        self,
        x,
        causal=False,
        mask=None,
        rotary_embed: Module | None = None,
        value_residual=None,
        return_values=False,
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # split heads

        q, k, v = map(self.split_heads, (q, k, v))

        # maybe query key norm

        if self.norm_qk:
            q, k = map(self.l2norm, (q, k))

        # scaling queries and keys - this would line up with the popular use of qk rmsnorm from google deepmind and now black forest labs - will use multihead rmsnorm

        q = q * rearrange(self.q_scale(), "(h d) -> h 1 d", h=self.heads)
        k = k * rearrange(self.k_scale(), "(h d) -> h 1 d", h=self.heads)

        # rotary positions

        if exists(rotary_embed):
            q = rotary_embed.rotate_queries_or_keys(q)
            k = rotary_embed.rotate_queries_or_keys(k)

        # for non-autoregressive masking

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")

        # maybe value residual, from resformer paper

        if exists(value_residual):
            v = v + value_residual

        # scale is sqrt(dk)

        with self.sdpa_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=causal, scale=self.attn_scale
            )

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_values:
            return out

        return out, v


# feedforward


class FeedForward(Module):
    def __init__(
        self,
        dim,
        *,
        expand_factor=4,
        manual_norm_weights=False,
        s_hidden_init=1.0,
        s_hidden_scale=1.0,
        s_gate_init=1.0,
        s_gate_scale=1.0,
        norm_eps=0.0,
        num_hyperspheres=1,
    ):
        super().__init__()
        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres,
        )

        self.dim = dim
        dim_inner = int(dim * expand_factor * 2 / 3)

        self.to_hidden = NormLinear_(dim, dim_inner)
        self.to_gate = NormLinear_(dim, dim_inner)

        self.hidden_scale = Scale(dim_inner, s_hidden_init, s_hidden_scale)
        self.gate_scale = Scale(dim_inner, s_gate_init, s_gate_scale)

        self.to_out = NormLinear_(dim_inner, dim, norm_dim_in=False)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale()
        gate = gate * self.gate_scale() * (self.dim**0.5)

        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)

class SinusoidalTimeEmbedding(Module):
    """Fourier features for time embedding"""
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0 # dim must be even
        self.dim = dim
        half_dim = dim // 2
        # From DiT/VDM - timestep embedding
        freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        # t: (B, 1)
        t_proj = t * self.freqs[None] * 2 * math.pi
        return torch.cat([t_proj.sin(), t_proj.cos()], dim=-1)

class TimestepEmbedder(Module):
    """Time embedding with Fourier features followed by MLP"""
    def __init__(self, dim, model_dim):
        super().__init__()
        self.fourier = SinusoidalTimeEmbedding(dim)
        # MLP to get conditioning vectors for AdaLN
        self.mlp = nn.Sequential(
            NormLinear(dim, dim * 2),
            nn.SiLU(),
            NormLinear(dim * 2, model_dim)
        )

    def forward(self, t):
        # t: (B, 1)
        t_fourier = self.fourier(t)
        return self.mlp(t_fourier)

class DiTBlock(Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, manual_norm_weights=False, norm_eps=0.0, num_hyperspheres=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

        self.attn = Attention(dim, heads=num_heads, manual_norm_weights=manual_norm_weights,
                            norm_eps=norm_eps, num_hyperspheres=num_hyperspheres)
        self.ff = FeedForward(dim, expand_factor=mlp_ratio, manual_norm_weights=manual_norm_weights,
                           norm_eps=norm_eps, num_hyperspheres=num_hyperspheres)

        # Zero-init modulation params using NormLinear
        self.modulation = NormLinear(
            dim,  # input dim from time embedding
            dim * 6,  # output dim for shift/scale/gate
            norm_dim_in=True,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres
        )

        # Zero init the weights while respecting hyperspherical constraints
        with torch.no_grad():
            # Initialize to zeros then normalize to maintain unit norm
            self.modulation.linear.weight.zero_()
            self.modulation.norm_weights_()

    def forward(self, x, c):
        # Get modulation params (now properly normalized)
        modulation = self.modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            modulation, 6, dim=-1
        )

        # Attention block with AdaLN
        x_norm = self.norm1(x)
        x_attn = x_norm * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = x + gate_msa[:, None] * self.attn(x_attn)

        # FF block with AdaLN
        x_norm = self.norm2(x)
        x_ff = x_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp[:, None] * self.ff(x_ff)

        return x


class nSFM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        beta=1.0,
        sigma=1.0,
        dim_head=64,
        heads=8,
        attn_norm_qk=True,
        ff_expand_factor=4.0,
        ce_ignore_index=-1,
        manual_norm_weights=False,
        tied_embedding=False,
        num_hyperspheres=1,
        causal=True,
        add_value_residual=True,
        alpha_init: (
            float | None
        ) = None,  # this would set the alpha init for all residuals, but would be overridden by alpha_attn_init and alpha_ff_init if they are specified
        s_logit_init: float = 1.0,
        s_logit_scale: float | None = None,
        alpha_attn_init: float | tuple[float, ...] | None = None,
        alpha_attn_scale: float | tuple[float, ...] | None = None,
        alpha_ff_init: float | tuple[float, ...] | None = None,
        alpha_ff_scale: float | tuple[float, ...] | None = None,
        s_qk_init: float | tuple[float, ...] = 1.0,
        s_qk_scale: float | tuple[float, ...] | None = None,
        s_ff_hidden_init: float | tuple[float, ...] = 1.0,
        s_ff_hidden_scale: float | tuple[float, ...] = 1.0,
        s_ff_gate_init: float | tuple[float, ...] = 1.0,
        s_ff_gate_scale: float | tuple[float, ...] = 1.0,
        attn_flash_kwargs: dict = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ),
        norm_eps=0.0,  # greater than 0 allows the norm to be around (1. - norm_eps) to (1. + norm_eps)
    ):
        super().__init__()
        self.beta = beta
        self.num_tokens = num_tokens
        self.dim = dim
        self.causal = causal
        self.add_value_residual = add_value_residual
        self.ignore_index = ce_ignore_index
        self.sigma = sigma

        # Initialize normalization and embedding layers
        NormLinear_ = partial(
            NormLinear,
            parametrize=not manual_norm_weights,
            norm_eps=norm_eps,
            groups=num_hyperspheres,
        )
        self.l2norm = partial(l2norm, norm_eps=norm_eps, groups=num_hyperspheres)
        self.token_embed = NormLinear_(dim, num_tokens)

        self.time_embed = TimestepEmbedder(256, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)

        # Set default alpha init
        alpha_init = default(alpha_init, 1.0 / depth)

        # Cast all scale hyperparameters to tuples of length depth
        scale_hparams = (
            alpha_attn_init,
            alpha_attn_scale,
            alpha_ff_init,
            alpha_ff_scale,
            s_qk_init,
            s_qk_scale,
            s_ff_hidden_init,
            s_ff_hidden_scale,
            s_ff_gate_init,
            s_ff_gate_scale,
        )
        scale_hparams = tuple(cast_tuple(hparam, depth) for hparam in scale_hparams)

        # Build transformer layers
        self.layers = ModuleList()
        for hparams in zip(*scale_hparams):

            dit_block = DiTBlock(
                dim=dim,
                num_heads=heads,
                mlp_ratio=ff_expand_factor
            )
            self.layers.append(dit_block)

        # Output layers
        self.to_logits = None if tied_embedding else NormLinear_(dim, num_tokens)
        self.logit_scale = Scale(
            num_tokens, s_logit_init, default(s_logit_scale, dim**-0.5)
        )

    def _process_tokens(self, tokens, t, beta_t):
        """Apply noise and normalization to tokens"""
        noise = torch.randn_like(tokens)
        noise = noise - (noise * tokens).sum(dim=-1, keepdim=True) * tokens
        tokens = self.l2norm(tokens + beta_t[:, None] * noise)
        return tokens#torch.cat([], dim=1)

    def _get_logits(self, tokens, token_embed):
        """Convert tokens to logits"""
        if exists(self.to_logits):
            logits = self.to_logits(tokens)
        else:
            logits = einsum(tokens, token_embed, "b n d, k d -> b n k")
        return logits * self.logit_scale()

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if isinstance(module, NormLinear):
                module.norm_weights_()

    def forward(self, ids, mask=None, return_loss=False):
        token_embed, rotary_embed = self.token_embed.weight, self.rotary_embed

        if return_loss:
            tokens = token_embed[ids]
            t = torch.rand(
                (tokens.size(0), 1), device=tokens.device, dtype=tokens.dtype
            )
            t_embed = self.time_embed(t)#.unsqueeze(1)
            beta_t = self.beta * (t**2)

            tokens = self._process_tokens(tokens, t, beta_t)
            first_values = None

            for block in self.layers:
                tokens = block(tokens, t_embed)


            tokens = tokens[:, :, :]
            logits = self._get_logits(tokens, token_embed)

            if not return_loss:
                return logits

            loss = F.cross_entropy(
                rearrange(logits, "b n c -> b c n"),
                ids,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            return (t[:, None] * loss).mean()

    @torch.inference_mode()
    def sample(
        self,
        batch_size=8,
        seq_len=1024,
        nb_steps=100,
        temperature=1.0,
        top_k=50,
        device=None,
    ):
        device = default(device, next(self.parameters()).device)
        dtype = next(self.parameters()).dtype

        self.eval()
        token_embed = self.token_embed.weight
        tokens = F.normalize(
            torch.randn(batch_size, seq_len, self.dim, device=device, dtype=dtype),
            dim=-1,
        )

        for i in range(1, nb_steps + 1):
            t = torch.full(
                (batch_size, 1), (nb_steps - i) / nb_steps, device=device, dtype=dtype
            )
            t_embed = self.time_embed(t)#.unsqueeze(1)

            x = tokens#torch.cat([t_embed, ], dim=1)
            first_values = None

            for block in self.layers:
                x = block(x, t_embed)

            x = x[:, :, :]
            logits = self._get_logits(x, token_embed) / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits = torch.where(logits < v[:, :, [-1]], -float("Inf"), logits)

            probs = F.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(
                probs.view(-1, self.num_tokens), num_samples=1
            ).view(batch_size, seq_len)
            tokens = token_embed[sampled_ids]

            if i < nb_steps:
                beta_t = self.beta * (t**2)
                tokens = self._process_tokens(tokens, t, beta_t)[:, :]

        similarities = einsum(tokens, token_embed, "b n d, k d -> b n k")
        return similarities.argmax(dim=-1)


if __name__ == "__main__":
    import torch
    from torch import optim
    from torch.utils.data import DataLoader, Dataset
    import numpy as np
    import matplotlib.pyplot as plt

    # PyTorch performance settings
    torch.backends.cudnn.benchmark = True

    # GPT-2 124M hyperparameters
    num_tokens = 32768  # Vocabulary size emu codebook
    dim = 768  # Hidden size
    depth = 12  # Number of transformer blocks
    heads = 12  # Number of attention heads
    dim_head = 64  # Dimension per head (dim // heads)
    seq_len = 1024  # Sequence length
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3

    # Instantiate the model
    model = nSFM(
        num_tokens=num_tokens,
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        causal=False,
        tied_embedding=True,
        beta=10.0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.to(torch.bfloat16)  # Convert model to bfloat16

    # Print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    class SimpleDataset(Dataset):
        def __init__(self, data_file="output_tokens.npy", block_size: int = 1024):
            super().__init__()
            self.data = np.load(data_file)
            self.length = self.data.shape[0]

            self.block_size = block_size

        def __getitem__(self, idx):
            return torch.from_numpy(self.data[idx, : self.block_size]).long()

        def __len__(self):
            return self.length

    dataset = SimpleDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    from tqdm.auto import tqdm

    loss_values = []  # For tracking loss
    N = 100  # Save generated tokens every N steps
    step = 0

    for epoch in range(num_epochs):
        for input_ids in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids = input_ids.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = model(input_ids, return_loss=True)

            loss.backward()
            optimizer.step()

            loss_values.append(loss.item())

            if step % N == 0:
                # Generate tokens with sampling
                start_tokens = input_ids[:, :1]
                generated_tokens = model.sample()

                generated_tokens = generated_tokens.detach().cpu().numpy()
                np.save("gen_tokens.npy", generated_tokens)

                # Plot loss (unchanged)
                plt.figure(figsize=(10, 5))
                plt.plot(loss_values)
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.title("Training Loss Over Time")
                plt.savefig("training_loss.png")
            step += 1

        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
