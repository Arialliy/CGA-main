"""Training-time target builders for CGA-v2.

All functions accept binary masks shaped [B, 1, H, W].  They are intentionally
simple and deterministic; no dataset-specific heuristics are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CGATargetConfig:
    center_radius: int = 2
    boundary_radius: int = 1
    peak_radius: int = 1
    max_scale_area: float = 512.0


def ensure_4d(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask[None, None]
    elif mask.dim() == 3:
        mask = mask[:, None]
    if mask.dim() != 4:
        raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(mask.shape)}")
    return mask.float()


def binary_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = ensure_4d(mask)
    if radius <= 0:
        return (mask > 0).float()
    k = 2 * int(radius) + 1
    return (F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=radius) > 0).float()


def binary_erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    mask = ensure_4d(mask)
    if radius <= 0:
        return (mask > 0).float()
    inv = 1.0 - (mask > 0).float()
    return 1.0 - binary_dilate(inv, radius)


def _connected_components_numpy(binary_2d: torch.Tensor):
    # Small dependency-free BFS over foreground pixels.  Used only for target building.
    arr = (binary_2d.detach().cpu().numpy() > 0).astype("uint8")
    h, w = arr.shape
    seen = [[False] * w for _ in range(h)]
    comps = []
    for y in range(h):
        for x in range(w):
            if arr[y, x] == 0 or seen[y][x]:
                continue
            stack = [(y, x)]
            seen[y][x] = True
            coords = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and arr[ny, nx] and not seen[ny][nx]:
                            seen[ny][nx] = True
                            stack.append((ny, nx))
            comps.append(coords)
    return comps


def build_center_and_scale(mask: torch.Tensor, cfg: CGATargetConfig) -> tuple[torch.Tensor, torch.Tensor]:
    mask = ensure_4d(mask)
    b, _, h, w = mask.shape
    center = torch.zeros_like(mask)
    scale = torch.zeros_like(mask)
    for bi in range(b):
        comps = _connected_components_numpy(mask[bi, 0] > 0)
        for comp in comps:
            if not comp:
                continue
            ys = torch.tensor([p[0] for p in comp], device=mask.device, dtype=torch.float32)
            xs = torch.tensor([p[1] for p in comp], device=mask.device, dtype=torch.float32)
            cy = int(torch.round(ys.mean()).clamp(0, h - 1).item())
            cx = int(torch.round(xs.mean()).clamp(0, w - 1).item())
            center[bi, 0, cy, cx] = 1.0
            area = float(len(comp))
            scale_value = min(1.0, area / max(float(cfg.max_scale_area), 1.0))
            for yy, xx in comp:
                scale[bi, 0, yy, xx] = scale_value
    center = binary_dilate(center, cfg.center_radius)
    return center, scale


def build_boundary(mask: torch.Tensor, cfg: CGATargetConfig) -> torch.Tensor:
    mask = ensure_4d(mask)
    fg = (mask > 0).float()
    outer = binary_dilate(fg, cfg.boundary_radius)
    inner = binary_erode(fg, cfg.boundary_radius)
    return ((outer - inner) > 0).float()


def build_peak(mask: torch.Tensor, center: torch.Tensor, cfg: CGATargetConfig) -> torch.Tensor:
    if cfg.peak_radius <= 0:
        return (center > 0).float()
    return binary_dilate(center, cfg.peak_radius)


def build_cga_targets(mask: torch.Tensor, cfg: CGATargetConfig | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or CGATargetConfig()
    mask = ensure_4d(mask)
    fg = (mask > 0).float()
    center, scale = build_center_and_scale(fg, cfg)
    boundary = build_boundary(fg, cfg)
    peak = build_peak(fg, center, cfg)
    return {
        "cga_center_target": center,
        "cga_boundary_target": boundary,
        "cga_scale_target": scale,
        "cga_peak_target": peak,
        "cga_foreground_target": fg,
    }


def summarize_cga_targets(targets: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k + "_sum": float(v.detach().sum().cpu().item()) for k, v in targets.items() if torch.is_tensor(v)}
