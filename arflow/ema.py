"""Exponential Moving Average module for improved inference stability."""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMAModel:
    """
    EMA model: ema_param = decay * ema_param + (1 - decay) * current_param.
    Supports delayed start and periodic updates.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        update_every: int = 1,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.update_step = 0

        self.shadow_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().to(device)

        self.collected_params = {}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Update EMA params. Decay ramps linearly from min_decay to decay."""
        if step < self.update_after_step:
            return

        if (step - self.update_after_step) % self.update_every != 0:
            return

        self.update_step += 1

        decay = self.decay
        if self.min_decay < self.decay:
            progress = min(1.0, (step - self.update_after_step) / 10000)
            decay = self.min_decay + progress * (self.decay - self.min_decay)

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].data.mul_(decay).add_(
                    param.data, alpha=1 - decay
                )

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA params to target model (for inference)."""
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].data)

    def store(self, model: nn.Module) -> None:
        """Store original model params (for restoring after training)."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.collected_params[name] = param.data.clone()

    def restore(self, model: nn.Module) -> None:
        """Restore original model params."""
        for name, param in model.named_parameters():
            if name in self.collected_params:
                param.data.copy_(self.collected_params[name])
        self.collected_params.clear()

    def state_dict(self) -> dict:
        return {
            'decay': self.decay,
            'min_decay': self.min_decay,
            'update_after_step': self.update_after_step,
            'update_every': self.update_every,
            'update_step': self.update_step,
            'shadow_params': self.shadow_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict.get('decay', self.decay)
        self.min_decay = state_dict.get('min_decay', self.min_decay)
        self.update_after_step = state_dict.get('update_after_step', self.update_after_step)
        self.update_every = state_dict.get('update_every', self.update_every)
        self.update_step = state_dict.get('update_step', 0)

        shadow_params = state_dict.get('shadow_params', {})
        for name, param in shadow_params.items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(param)


def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float
) -> None:
    """Simple EMA update function (functional API)."""
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
