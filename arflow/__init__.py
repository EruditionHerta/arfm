"""AR-FlowMatching: autoregressive flow matching with U-Net + SPADE."""

from .time_field import SigmoidTimeField
from .model import (
    ARFlowSPADEUNet,
    ARFlowMatching,
    SPADE,
    SPADEResBlock,
    SelfAttention2d,
    MultiHeadAttentionBlock,
    SinusoidalEmbedding,
    TimestepEmbedder,
    LabelEmbedding,
)
from .solver import EulerSampler, RK4Sampler, HeunSampler, ConditionalEulerSampler
from .solver import compute_ar_order_metric
from .ema import EMAModel, update_ema
from .utils import (
    get_timestamped_output_dir,
    save_args_to_txt,
    zero_module,
    scale_module,
    checkpoint,
    normalization,
    timestep_embedding
)

__all__ = [
    'SigmoidTimeField',
    'LinearTimeField',
    'ARFlowSPADEUNet',
    'ARFlowUNet',
    'ARFlowMatching',
    'SPADE',
    'SPADEResBlock',
    'SelfAttention2d',
    'MultiHeadAttentionBlock',
    'SinusoidalEmbedding',
    'TimestepEmbedder',
    'LabelEmbedding',
    'EulerSampler',
    'RK4Sampler',
    'HeunSampler',
    'ConditionalEulerSampler',
    'compute_ar_order_metric',
    'EMAModel',
    'update_ema',
    'get_timestamped_output_dir',
    'save_args_to_txt',
    'zero_module',
    'scale_module',
    'checkpoint',
    'normalization',
    'timestep_embedding'
]
