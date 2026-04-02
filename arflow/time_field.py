"""
Spatially heterogeneous adaptive AR time field for center-to-edge autoregressive generation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AdaptiveARTimeField:
    """
    AR time field: tau=0 is pure noise, tau=1 is fully clear.
    Center generates first, edges follow; early phase: center fast/edge slow,
    late phase: center saturates while edges catch up.

    Formula: tau_start(r) = start_delay * r^power, normalized_tau = (tau - tau_start) / duration
    """

    def __init__(
        self,
        start_delay: float = 0.3,
        power: float = 2.0,
        k: float = 2.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.start_delay = start_delay
        self.power = power
        self.k = k
        self.device = device
        self._distance_map = None
        self._last_shape = None

    def _compute_ring_distance_map(self, H: int, W: int) -> torch.Tensor:
        """Compute normalized Euclidean distance map [H, W], center=0 edge=1."""
        cy, cx = (H - 1) / 2, (W - 1) / 2

        y_coords = torch.arange(H, device=self.device, dtype=torch.float32)
        x_coords = torch.arange(W, device=self.device, dtype=torch.float32)
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')

        dist_y = Y - cy
        dist_x = X - cx
        ring_dist = torch.sqrt(dist_y**2 + dist_x**2)

        max_ring = ring_dist.max()
        if max_ring > 0:
            r_map = ring_dist / max_ring
        else:
            r_map = ring_dist

        return r_map

    def get_time_map(
        self,
        tau: torch.Tensor,
        shape: Tuple[int, int, int, int]  # [B, C, H, W]
    ) -> torch.Tensor:
        """Compute spatial time map from global time tau, returns [B, 1, H, W]."""
        B, C, H, W = shape

        if isinstance(tau, torch.Tensor):
            device = tau.device
        else:
            device = self.device

        if self._distance_map is None or self._last_shape != (H, W) or self._distance_map.device != device:
            self._distance_map = self._compute_ring_distance_map(H, W).to(device)
            self._last_shape = (H, W)

        r_map = self._distance_map

        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor([tau], device=device)
        elif tau.dim() == 0:
            tau = tau.unsqueeze(0)

        if tau.shape[0] == 1 and B > 1:
            tau = tau.expand(B)

        # Wave diffusion: center completes first and holds, edges complete at tau=1
        tau_expanded = tau.view(-1, 1, 1)  # [B, 1, 1]
        r_expanded = r_map.unsqueeze(0)     # [1, H, W]
        tau_start = self.start_delay * (r_expanded ** self.power)

        duration = max(1.0 - self.start_delay, 1e-6)
        normalized_tau = (tau_expanded - tau_start) / duration

        t_map = torch.clamp(normalized_tau, 0.0, 1.0)

        return t_map.unsqueeze(1)  # [B, 1, H, W]

    def get_time_delta(
        self,
        tau_curr: torch.Tensor,
        tau_next: torch.Tensor,
        shape: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """Compute time delta map between two global times for ODE solver updates."""
        t_curr = self.get_time_map(tau_curr, shape)
        t_next = self.get_time_map(tau_next, shape)
        return t_next - t_curr

    def get_time_weight(
        self,
        tau: torch.Tensor,
        shape: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        """
        Compute derivative dt_p/dtau as unbiased weight for Flow Matching Loss.
        Non-evolved or completed regions naturally have zero derivative, no hard cutoff needed.
        """
        B, C, H, W = shape

        if isinstance(tau, torch.Tensor):
            device = tau.device
        else:
            device = self.device

        if self._distance_map is None or self._last_shape != (H, W) or self._distance_map.device != device:
            self._distance_map = self._compute_ring_distance_map(H, W).to(device)
            self._last_shape = (H, W)

        r_map = self._distance_map

        if not isinstance(tau, torch.Tensor):
            tau = torch.tensor([tau], device=device)
        elif tau.dim() == 0:
            tau = tau.unsqueeze(0)

        if tau.shape[0] == 1 and B > 1:
            tau = tau.expand(B)

        tau_expanded = tau.view(-1, 1, 1)
        r_expanded = r_map.unsqueeze(0)

        tau_start = self.start_delay * (r_expanded ** self.power)

        n_raw = (tau_expanded - tau_start) / (1.0 - tau_start + 1e-8)

        # Derivative is non-zero only within the evolution interval (0 < n_raw < 1)
        valid_mask = (n_raw > 0) & (n_raw < 1)

        # Clamped linear method: dt_p/dn = 1.0
        dt_dn = 1.0

        # Chain rule: dt_p/dtau = (dt_p/dn) * (dn/dtau)
        duration = max(1.0 - self.start_delay, 1e-6)
        dn_dtau = 1.0 / duration
        dt_dtau = dt_dn * dn_dtau

        weight = dt_dtau * valid_mask.float()

        return weight.unsqueeze(1)  # [B, 1, H, W]

    def visualize_time_progression(
        self,
        num_steps: int = 10
    ) -> torch.Tensor:
        """Visualize time field evolution over global time, returns [num_steps, H, W]."""
        H, W = 28, 28
        taus = torch.linspace(0, 1, num_steps, device=self.device)

        time_maps = []
        for tau in taus:
            t_map = self.get_time_map(tau, (1, 1, H, W))
            time_maps.append(t_map.squeeze(1))

        return torch.stack(time_maps)


def compute_spatial_timesteps(t_map: torch.Tensor) -> torch.Tensor:
    """Downsample pixel-level time map to patch level [B, num_patches]."""
    B, C, H, W = t_map.shape

    patch_size = 16
    h_patches = H // patch_size
    w_patches = W // patch_size

    t_patched = torch.nn.functional.adaptive_avg_pool2d(
        t_map, (h_patches, w_patches)
    )

    return t_patched.reshape(B, -1)


# Backward-compatible alias
SigmoidTimeField = AdaptiveARTimeField


if __name__ == "__main__":
    time_field = AdaptiveARTimeField(start_delay=0.3, power=2.0, k=8.0)
    time_maps = time_field.visualize_time_progression(num_steps=10)
    print(f"Time field sequence shape: {time_maps.shape}")
