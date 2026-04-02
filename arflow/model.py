"""
AR-FlowMatching: U-Net + SPADE architecture for spatially-heterogeneous time fields.
Supports label-conditional generation and CFG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from .time_field import AdaptiveARTimeField
from .utils import zero_module


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal position embedding"""

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * self.scale * emb
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimestepEmbedder(nn.Module):
    """Scalar timestep -> embedding vector via sinusoidal encoding + MLP"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

    @staticmethod
    def timestep_embedding(timesteps, dim, max_period=10000):
        """Create sinusoidal timestep embedding"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedding(nn.Module):
    """Label embedding module"""

    def __init__(self, num_classes: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding(labels)


class SPADE(nn.Module):
    """Spatially-Adaptive Normalization for injecting spatially-heterogeneous time field."""

    def __init__(self, norm_nc: int, cond_nc: int = 1, hidden_nc: int = 128, tau_emb_dim: int = 0):
        super().__init__()

        self.param_free_norm = nn.GroupNorm(32, norm_nc, affine=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(cond_nc, hidden_nc, kernel_size=3, padding=1),
            nn.SiLU()
        )

        self.tau_emb_dim = tau_emb_dim if tau_emb_dim is not None else 0
        if self.tau_emb_dim > 0:
            self.tau_proj = nn.Conv2d(tau_emb_dim, hidden_nc, 1)
        else:
            self.tau_proj = None

        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

        # Zero init: at training start SPADE acts as identity, critical for stability
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_map: torch.Tensor,
        tau_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape

        normalized = self.param_free_norm(x)

        # Must use bilinear to maintain smooth spatial gradients in t_map
        if t_map.shape[2:] != (H, W):
            t_map = F.interpolate(t_map, size=(H, W), mode='bilinear', align_corners=False)

        actv = self.mlp_shared(t_map)

        if tau_emb is not None and self.tau_proj is not None:
            tau_broadcast = tau_emb.expand(-1, -1, H, W)
            actv = actv + self.tau_proj(tau_broadcast)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return normalized * (1 + gamma) + beta


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention (ref: TorchCFM). Output proj zero-initialized."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels

        assert channels % num_heads == 0, f"channels={channels} must be divisible by num_heads={num_heads}"
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)

        qkv = self.qkv(h).view(B, 3, self.num_heads, self.head_dim, -1)
        q, k, v = qkv.unbind(1)

        scale = self.head_dim ** -0.5
        attn_input = torch.einsum('bhci,bhcj->bhij', q, k) * scale
        # Numerical stabilization: subtract max before softmax to prevent overflow
        attn_input = attn_input - attn_input.amax(dim=3, keepdim=True)
        attn = torch.softmax(attn_input, dim=3)

        out = torch.einsum('bhij,bhcj->bhci', attn, v)
        out = out.reshape(B, C, H, W)

        return x + self.proj(out)


class SelfAttention2d(MultiHeadAttentionBlock):
    """Single-head self-attention (backward compatible alias)."""

    def __init__(self, channels: int):
        super().__init__(channels, num_heads=1)


class SPADEResBlock(nn.Module):
    """Residual block with SPADE: x -> SPADE1 -> SiLU -> Conv3x3 -> SPADE2 -> SiLU -> Conv3x3 + Skip"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_nc: int = 128,
        use_sa: bool = False,
        tau_emb_dim: int = 0,
        dropout: float = 0.0
    ):
        super().__init__()

        self.spade1 = SPADE(in_channels, cond_nc=1, hidden_nc=hidden_nc, tau_emb_dim=tau_emb_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.spade2 = SPADE(out_channels, cond_nc=1, hidden_nc=hidden_nc, tau_emb_dim=tau_emb_dim)
        # Zero init ensures the entire block is initially an identity mapping
        self.conv2 = zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))

        self.act = nn.SiLU()

        self.dropout = dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout2d(p=dropout)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

        self.use_sa = use_sa
        if use_sa:
            self.sa = SelfAttention2d(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        t_map: torch.Tensor,
        tau_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.spade1(x, t_map, tau_emb)
        h = self.act(h)
        h = self.conv1(h)

        h = self.spade2(h, t_map, tau_emb)
        h = self.act(h)
        h = self.conv2(h)

        if self.dropout > 0 and self.training:
            h = self.dropout_layer(h)

        if self.use_sa:
            h = self.sa(h)

        return h + self.skip(x)


class ARFlowSPADEUNet(nn.Module):
    """U-Net + SPADE backbone for AR-FlowMatching. Supports multi-head attention, dropout, label conditioning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8, 16),
        num_res_blocks: int = 2,
        spade_hidden_nc: int = 128,
        num_classes: int = 0,
        use_cfg: bool = False,
        cfg_drop_prob: float = 0.1,
        label_embed_dim: int = 128,
        time_embed_dim: int = None,
        attention_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_cfg = use_cfg
        self.cfg_drop_prob = cfg_drop_prob
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else base_channels * 4
        self.attention_heads = attention_heads
        self.dropout = dropout

        self.time_encoder = nn.Sequential(
            SinusoidalEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim)
        )

        if num_classes > 0:
            self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
            if use_cfg:
                self.null_embedding = nn.Parameter(torch.zeros(label_embed_dim))
            else:
                self.register_buffer('null_embedding', torch.zeros(label_embed_dim))

            self.cond_fusion = nn.Linear(self.time_embed_dim + label_embed_dim, self.time_embed_dim)
        else:
            self.label_embedding = None
            self.register_buffer('null_embedding', torch.zeros(0))
            self.cond_fusion = nn.Identity()

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()

        channels = [base_channels * c for c in channel_mult]
        for i, (in_ch, out_ch) in enumerate(zip([base_channels] + channels[:-1], channels)):
            blocks = nn.ModuleList([
                SPADEResBlock(
                    in_ch if j == 0 else out_ch,
                    out_ch,
                    spade_hidden_nc,
                    use_sa=(i >= 2),  # Attention in deeper layers only
                    tau_emb_dim=time_embed_dim,
                    dropout=dropout
                )
                for j in range(num_res_blocks)
            ])
            self.encoder_blocks.append(blocks)

            if i < len(channels) - 1:
                self.encoder_pools.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

        self.mid_blocks = nn.ModuleList([
            SPADEResBlock(channels[-1], channels[-1], spade_hidden_nc, use_sa=True, tau_emb_dim=time_embed_dim, dropout=dropout)
            for _ in range(2)
        ])

        self.decoder_blocks = nn.ModuleList()

        # Skip channels from deep to shallow, excluding bottleneck (channels[-1])
        # e.g. channels = [64,128,256,512,1024] -> decoder_skip = [512,256,128,64]
        num_decoder_layers = len(channels) - 1
        decoder_skip_channels = channels[-2::-1]

        for i in range(num_decoder_layers):
            skip_ch = decoder_skip_channels[i]
            out_ch = skip_ch

            if i == 0:
                block_in_ch = channels[-1] + skip_ch
            else:
                block_in_ch = decoder_skip_channels[i - 1] + skip_ch

            blocks = nn.ModuleList([
                SPADEResBlock(
                    block_in_ch if j == 0 else out_ch,
                    out_ch,
                    spade_hidden_nc,
                    use_sa=True,
                    tau_emb_dim=time_embed_dim,
                    dropout=dropout
                )
                for j in range(num_res_blocks)
            ])
            self.decoder_blocks.append(blocks)

        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        """DDPM-style Xavier init (gain=2^0.5), output layer zero init."""

        def _init(module):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=2**0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)

        if self.label_embedding is not None:
            nn.init.normal_(self.label_embedding.weight, std=0.02)

        # Zero init output layer so initial predictions are ~0
        nn.init.zeros_(self.conv_out.weight)
        if self.conv_out.bias is not None:
            nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        t_map: torch.Tensor,
        tau: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        label_emb = None
        if self.label_embedding is not None:
            if labels is None:
                label_emb = self.null_embedding.unsqueeze(0).expand(B, -1)
            else:
                if labels.dim() == 2:  # Multi-label (CelebA): [B, num_classes] matrix multiply
                    label_emb = labels.to(self.label_embedding.weight.dtype) @ self.label_embedding.weight
                else:
                    label_emb = self.label_embedding(labels)

                # CFG: randomly replace label embeddings with null during training
                if self.use_cfg and self.training:
                    mask = torch.rand(B, device=device) < self.cfg_drop_prob
                    if mask.any():
                        null_emb = self.null_embedding.unsqueeze(0).expand(B, -1)
                        label_emb = torch.where(mask.unsqueeze(1), label_emb, null_emb)

        # Ensure tau is [B]-shaped
        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor([tau], device=device, dtype=torch.float32)
        if tau.dim() == 0:
            tau = tau.unsqueeze(0)
        if tau.shape[0] == 1 and B > 1:
            tau = tau.expand(B)

        time_emb = self.time_encoder(tau * 1000.0)

        if label_emb is not None:
            cond_emb = self.cond_fusion(torch.cat([time_emb, label_emb], dim=1))
        else:
            cond_emb = time_emb

        tau_emb = cond_emb.unsqueeze(-1).unsqueeze(-1)  # [B, D, 1, 1] for SPADE

        x = self.conv_in(x)

        # Encoder: collect skip connections
        skip_connections = []
        for i, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                x = block(x, t_map, tau_emb)
            skip_connections.append(x.clone())

            if i < len(self.encoder_pools):
                x = self.encoder_pools[i](x)

        # Bottleneck
        for block in self.mid_blocks:
            x = block(x, t_map, tau_emb)

        # Decoder: skip from deepest to shallowest, excluding bottleneck
        # skip_connections[-1] is bottleneck output, so start from [-2]
        for i, blocks in enumerate(self.decoder_blocks):
            skip_idx = len(skip_connections) - 2 - i
            skip = skip_connections[skip_idx]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')

            x = torch.cat([x, skip], dim=1)

            for block in blocks:
                x = block(x, t_map, tau_emb)

        return self.conv_out(x)


class ARFlowMatching(nn.Module):
    """AR-FlowMatching: wraps AdaptiveARTimeField + ARFlowSPADEUNet with flow matching loss (derivative-weighted MSE)."""

    def __init__(
        self,
        backbone: str = 'spade_unet',
        time_start_delay: float = 0.3,
        time_power: float = 2.0,
        time_k: float = 8.0,
        num_classes: int = 0,
        use_cfg: bool = False,
        cfg_drop_prob: float = 0.1,
        **backbone_kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_cfg = use_cfg

        self.time_field = AdaptiveARTimeField(
            start_delay=time_start_delay,
            power=time_power,
            k=time_k
        )

        backbone_kwargs['num_classes'] = num_classes
        backbone_kwargs['use_cfg'] = use_cfg
        backbone_kwargs['cfg_drop_prob'] = cfg_drop_prob

        if backbone == 'spade_unet':
            self.network = ARFlowSPADEUNet(**backbone_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'spade_unet'.")

    def forward(
        self,
        x_tau: torch.Tensor,
        tau: torch.Tensor,
        labels: torch.Tensor = None
    ) -> torch.Tensor:
        t_map = self.time_field.get_time_map(tau, x_tau.shape)
        return self.network(x_tau, t_map, tau=tau, labels=labels)

    def get_loss(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        tau: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """Flow matching loss with derivative weighting dt_p/dtau (mathematically unbiased by change-of-variables)."""
        B, C, H, W = x_0.shape

        with torch.no_grad():
            t_map = self.time_field.get_time_map(tau, x_0.shape)
            weight = self.time_field.get_time_weight(tau, x_0.shape).expand(B, C, H, W)

        t_expanded = t_map.expand(B, C, H, W)
        x_tau = (1 - t_expanded) * x_0 + t_expanded * x_1

        v_target = x_1 - x_0
        v_pred = self.network(x_tau, t_map, tau=tau, labels=labels)

        # Clamp diff to prevent NaN from extreme squared values
        diff = torch.clamp(v_pred - v_target, -100.0, 100.0)
        loss = (diff ** 2 * weight).mean()

        info = {
            'loss': loss.item(),
            'active_ratio': (weight > 1e-4).float().mean().item(),
            'mean_t': t_map.mean().item()
        }

        return loss, info


if __name__ == "__main__":
    model = ARFlowMatching(
        backbone='spade_unet',
        in_channels=1,
        base_channels=64,
        channel_mult=(1, 2, 4, 8, 16),
        num_res_blocks=2,
        spade_hidden_nc=128,
        num_classes=10,
        use_cfg=True
    )

    B, C, H, W = 4, 1, 28, 28
    x_0 = torch.randn(B, C, H, W)
    x_1 = torch.randn(B, C, H, W)
    tau = torch.rand(B)
    labels = torch.randint(0, 10, (B,))

    loss, info = model.get_loss(x_0, x_1, tau, labels=labels)
    print(f"Loss: {loss:.4f}, Info: {info}")

    v_pred = model(x_0, tau, labels=labels)
    print(f"Output shape: {v_pred.shape}, Params: {sum(p.numel() for p in model.parameters()):,}")
