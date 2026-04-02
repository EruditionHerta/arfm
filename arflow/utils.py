"""ARFM utility functions."""

import sys
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Callable, Any, Tuple


def get_timestamped_output_dir(base_dir, name_prefix, resume_path=None):
    """Generate timestamped output directory; reuse original dir if resume_path given."""
    if resume_path:
        resume_dir = Path(resume_path).parent
        resume_dir.mkdir(parents=True, exist_ok=True)
        return resume_dir

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    timestamped_dir = Path(base_dir) / f"{name_prefix}_{timestamp}"

    counter = 1
    while timestamped_dir.exists():
        timestamped_dir = Path(base_dir) / f"{name_prefix}_{timestamp}_{counter}"
        counter += 1

    timestamped_dir.mkdir(parents=True, exist_ok=True)
    return timestamped_dir


def save_args_to_txt(args, filepath, script_name="ARFM"):
    """Save hyperparameters to txt file."""
    filepath = Path(filepath)

    if hasattr(sys, 'argv'):
        command = ' '.join(sys.argv)
    else:
        command = 'unknown'

    lines = [
        f"# {script_name} parameter record",
        f"# Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Execution command: python {command}",
        "",
    ]

    args_dict = vars(args)
    for key in sorted(args_dict.keys()):
        value = args_dict[key]

        if isinstance(value, (list, tuple)):
            value_str = ' '.join(str(v) for v in value)
        else:
            value_str = str(value)

        lines.append(f"{key}: {value_str}")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize all parameters."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module: nn.Module, scale: float) -> nn.Module:
    """Scale all parameters."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def checkpoint(
    func: Callable,
    args: Tuple[Any, ...],
    flag: bool = False
) -> Any:
    """Gradient checkpointing to save memory."""
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *args)
    else:
        return func(*args)


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create convolution layer of arbitrary dimension."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Linear:
    """Create linear layer."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    """Create average pooling layer of arbitrary dimension."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ZeroConv2d(nn.Conv2d):
    """Zero-initialized 2D convolution layer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        zero_module(self)


def normalization(channels: int) -> nn.Module:
    """GroupNorm with 32 groups (diffusion model standard)."""
    return nn.GroupNorm(32, channels)


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: float = 10000
) -> torch.Tensor:
    """Sinusoidal timestep embedding (DDPM/TorchCFM standard)."""
    import math

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
