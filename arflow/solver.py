"""ODE solvers for AR-FlowMatching."""

import torch
import torch.nn as nn
from typing import Optional, Callable, List, Tuple
from .model import ARFlowMatching
from .time_field import SigmoidTimeField


class EulerSampler:
    """Euler method ODE solver with built-in value clamping (+/-3.0)."""

    CLAMP_VALUE = 3.0

    def __init__(
        self,
        model: ARFlowMatching,
        num_steps: int = 20,
        time_schedule: str = 'linear'
    ):
        self.model = model
        self.num_steps = num_steps
        self.time_schedule = time_schedule

    def get_time_schedule(self, device: torch.device) -> torch.Tensor:
        """Generate global time sequence."""
        if self.time_schedule == 'linear':
            return torch.linspace(0, 1, self.num_steps + 1, device=device)
        elif self.time_schedule == 'sigmoid':
            t = torch.linspace(0, 1, self.num_steps + 1, device=device)
            return torch.sigmoid(4 * (t - 0.5))
        else:
            raise ValueError(f"Unknown time_schedule: {self.time_schedule}")

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        condition_fn: Optional[Callable] = None,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample from pure noise."""
        self.model.eval()

        device = next(self.model.parameters()).device
        B, C, H, W = shape

        x = torch.randn(shape, device=device)
        tau_schedule = self.get_time_schedule(device)

        trajectory = [] if return_trajectory else None

        for i in range(self.num_steps):
            tau_curr = tau_schedule[i]
            tau_next = tau_schedule[i + 1]

            dT_map = self.model.time_field.get_time_delta(
                tau_curr, tau_next, x.shape
            )

            v_pred = self.model(x, tau_curr)

            # Each pixel uses different dt based on spatially heterogeneous time field
            dT_expanded = dT_map.expand(B, C, H, W)
            x = x + v_pred * dT_expanded
            x = torch.clamp(x, -self.CLAMP_VALUE, self.CLAMP_VALUE)

            if condition_fn is not None:
                x = condition_fn(x, tau_next)

            if return_trajectory:
                trajectory.append(x.cpu().clone())

        self.model.train()

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_intermediate(
        self,
        shape: Tuple[int, int, int, int],
        num_intermediate: int = 5
    ) -> List[torch.Tensor]:
        """Sample and return intermediate images at uniform intervals."""
        device = next(self.model.parameters()).device
        B, C, H, W = shape

        x = torch.randn(shape, device=device)
        tau_schedule = self.get_time_schedule(device)

        images = []
        save_indices = torch.linspace(0, self.num_steps, num_intermediate, dtype=torch.long)

        for i in range(self.num_steps + 1):
            if i in save_indices:
                if i == self.num_steps:
                    images.append(x.clone())
                else:
                    images.append(x.cpu().clone())

            if i >= self.num_steps:
                break

            tau_curr = tau_schedule[i]
            tau_next = tau_schedule[i + 1]

            dT_map = self.model.time_field.get_time_delta(tau_curr, tau_next, x.shape)
            v_pred = self.model(x, tau_curr)

            dT_expanded = dT_map.expand(B, C, H, W)
            x = x + v_pred * dT_expanded
            x = torch.clamp(x, -self.CLAMP_VALUE, self.CLAMP_VALUE)

        return images


class RK4Sampler:
    """4th-order Runge-Kutta solver (more accurate, more expensive)."""

    def __init__(
        self,
        model: ARFlowMatching,
        num_steps: int = 20
    ):
        self.model = model
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        condition_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Sample using RK4 method."""
        self.model.eval()
        device = next(self.model.parameters()).device
        B, C, H, W = shape

        x = torch.randn(shape, device=device)
        tau_schedule = torch.linspace(0, 1, self.num_steps + 1, device=device)

        for i in range(self.num_steps):
            tau_curr = tau_schedule[i]
            h = tau_schedule[i + 1] - tau_curr

            # Simplified RK4: uses uniform global time steps, not spatially heterogeneous
            k1 = self.model(x, tau_curr)

            x2 = x + 0.5 * h * k1
            k2 = self.model(x2, tau_curr + 0.5 * h)

            x3 = x + 0.5 * h * k2
            k3 = self.model(x3, tau_curr + 0.5 * h)

            x4 = x + h * k3
            k4 = self.model(x4, tau_curr + h)

            x = x + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

            if condition_fn is not None:
                x = condition_fn(x, tau_schedule[i + 1])

        self.model.train()
        return x


class HeunSampler:
    """Heun method (improved Euler, more accurate than standard Euler)."""

    def __init__(
        self,
        model: ARFlowMatching,
        num_steps: int = 20
    ):
        self.model = model
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        condition_fn: Optional[Callable] = None
    ) -> torch.Tensor:
        """Sample using Heun method."""
        self.model.eval()
        device = next(self.model.parameters()).device
        B, C, H, W = shape

        x = torch.randn(shape, device=device)
        tau_schedule = torch.linspace(0, 1, self.num_steps + 1, device=device)

        for i in range(self.num_steps):
            tau_curr = tau_schedule[i]
            tau_next = tau_schedule[i + 1]

            v1 = self.model(x, tau_curr)
            dT_map = self.model.time_field.get_time_delta(tau_curr, tau_next, x.shape)
            dT_expanded = dT_map.expand(B, C, H, W)
            x_pred = x + v1 * dT_expanded

            v2 = self.model(x_pred, tau_next)

            x = x + 0.5 * (v1 + v2) * dT_expanded

            if condition_fn is not None:
                x = condition_fn(x, tau_next)

        self.model.train()
        return x


class ConditionalEulerSampler(EulerSampler):
    """Conditional Euler sampler with label conditioning and CFG support."""

    def __init__(
        self,
        model: ARFlowMatching,
        num_steps: int = 20,
        time_schedule: str = 'linear'
    ):
        super().__init__(model, num_steps, time_schedule)

    @torch.no_grad()
    def sample_with_labels(
        self,
        shape: Tuple[int, int, int, int],
        labels: torch.Tensor,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample with label condition."""
        self.model.eval()
        device = next(self.model.parameters()).device
        B, C, H, W = shape

        x = torch.randn(shape, device=device)
        tau_schedule = self.get_time_schedule(device)

        trajectory = [] if return_trajectory else None

        for i in range(self.num_steps):
            tau_curr = tau_schedule[i]
            tau_next = tau_schedule[i + 1]

            dT_map = self.model.time_field.get_time_delta(tau_curr, tau_next, x.shape)

            v_pred = self.model(x, tau_curr, labels=labels)

            dT_expanded = dT_map.expand(B, C, H, W)
            x = x + v_pred * dT_expanded
            x = torch.clamp(x, -self.CLAMP_VALUE, self.CLAMP_VALUE)

            if return_trajectory:
                trajectory.append(x.cpu().clone())

        self.model.train()

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def sample_with_cfg(
        self,
        shape: Tuple[int, int, int, int],
        labels: torch.Tensor,
        cfg_scale: float = 1.5,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """Sample with Classifier-Free Guidance: v = v_uncond + scale * (v_cond - v_uncond)."""
        self.model.eval()
        device = next(self.model.parameters()).device
        B, C, H, W = shape

        x = torch.randn(shape, device=device)
        tau_schedule = self.get_time_schedule(device)

        trajectory = [] if return_trajectory else None

        for i in range(self.num_steps):
            tau_curr = tau_schedule[i]
            tau_next = tau_schedule[i + 1]

            dT_map = self.model.time_field.get_time_delta(tau_curr, tau_next, x.shape)

            v_uncond = self.model(x, tau_curr, labels=None)
            v_cond = self.model(x, tau_curr, labels=labels)

            v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)

            dT_expanded = dT_map.expand(B, C, H, W)
            x = x + v_pred * dT_expanded
            x = torch.clamp(x, -self.CLAMP_VALUE, self.CLAMP_VALUE)

            if return_trajectory:
                trajectory.append(x.cpu().clone())

        self.model.train()

        if return_trajectory:
            return x, trajectory
        return x


def compute_ar_order_metric(
    trajectory: List[torch.Tensor],
    center_ratio: float = 0.3
) -> dict:
    """Compute AR order metric: verify center becomes clear earlier than edge."""
    num_steps = len(trajectory)
    B, C, H, W = trajectory[0].shape

    center_h = int(H * center_ratio)
    center_w = int(W * center_ratio)
    h_start = (H - center_h) // 2
    h_end = h_start + center_h
    w_start = (W - center_w) // 2
    w_end = w_start + center_w

    center_clarity = []
    edge_clarity = []

    for x in trajectory:
        center = x[:, :, h_start:h_end, w_start:w_end]
        edge = x[:, :, 0:H // 4, 0:W // 4]

        center_clarity.append(center.std().item())
        edge_clarity.append(edge.std().item())

    center_leads = sum(c > e for c, e in zip(center_clarity, edge_clarity))
    ar_order_score = center_leads / num_steps

    return {
        'ar_order_score': ar_order_score,
        'center_clarity_trajectory': center_clarity,
        'edge_clarity_trajectory': edge_clarity
    }
