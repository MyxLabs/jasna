# SPDX-License-Identifier: AGPL-3.0
"""MUSA-compatible deformable convolution for BasicVSR++.

Some MUSA torchvision wheels expose deform_conv2d but fail at kernel launch
with "invalid device function". The compatibility path below implements the
torchvision DCNv2 definition with grid_sample and 1x1 grouped convolutions.
Only MUSA tensors use it by default. Other platforms retain the native op.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair


def deform_conv2d_grid_sample(
    input: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """DeformConv2d with interleaved (dy, dx) offsets and optional modulating mask.

    Streaming one kernel position at a time reduces peak activation memory.
    Handles deformable offset groups and ordinary convolution groups independently.
    """
    if input.ndim != 4 or offset.ndim != 4 or weight.ndim != 4:
        raise ValueError("input, offset, and weight must be 4D tensors")
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    n, cin, hin, win = input.shape
    cout, weight_cin, kh, kw = weight.shape
    k = kh * kw
    if offset.shape[0] != n or offset.shape[1] % (2 * k) != 0:
        raise ValueError("invalid offset batch or channel count")
    offset_groups = offset.shape[1] // (2 * k)
    if offset_groups < 1 or cin % offset_groups:
        raise ValueError("input channels must be divisible by offset groups")
    if weight_cin < 1 or cin % weight_cin:
        raise ValueError("invalid convolution groups")
    groups = cin // weight_cin
    if cout % groups:
        raise ValueError("output channels must be divisible by groups")
    hout, wout = offset.shape[2:]
    expected_h = (hin + 2 * pad_h - dil_h * (kh - 1) - 1) // stride_h + 1
    expected_w = (win + 2 * pad_w - dil_w * (kw - 1) - 1) // stride_w + 1
    if (hout, wout) != (expected_h, expected_w):
        raise ValueError("offset spatial size differs from convolution output")
    if mask is not None and tuple(mask.shape) != (n, offset_groups * k, hout, wout):
        raise ValueError("invalid DCNv2 mask shape")

    c_per_offset = cin // offset_groups
    grouped_input = input.reshape(n * offset_groups, c_per_offset, hin, win)
    grouped_offset = offset.reshape(n, offset_groups, k, 2, hout, wout)
    grouped_mask = mask.reshape(n, offset_groups, k, hout, wout) if mask is not None else None
    grid_dtype = torch.float32 if input.dtype in (torch.float16, torch.bfloat16) else input.dtype
    ys = torch.arange(hout, device=input.device, dtype=grid_dtype).reshape(1, 1, hout, 1)
    xs = torch.arange(wout, device=input.device, dtype=grid_dtype).reshape(1, 1, 1, wout)
    output = None
    for ky in range(kh):
        for kx in range(kw):
            idx = ky * kw + kx
            dy = grouped_offset[:, :, idx, 0].to(dtype=grid_dtype)
            dx = grouped_offset[:, :, idx, 1].to(dtype=grid_dtype)
            sample_y = ys * stride_h - pad_h + ky * dil_h + dy
            sample_x = xs * stride_w - pad_w + kx * dil_w + dx
            # align_corners=False works also for images with H or W equal to 1.
            grid = torch.stack((2 * (sample_x + 0.5) / win - 1,
                                2 * (sample_y + 0.5) / hin - 1), dim=-1)
            grid = grid.reshape(n * offset_groups, hout, wout, 2)
            sampled = F.grid_sample(grouped_input, grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            if grouped_mask is not None:
                sampled = sampled.reshape(n, offset_groups, c_per_offset, hout, wout)
                sampled = sampled * grouped_mask[:, :, idx].unsqueeze(2)
            sampled = sampled.reshape(n, cin, hout, wout)
            point_weight = weight[:, :, ky, kx].unsqueeze(-1).unsqueeze(-1)
            contribution = F.conv2d(sampled, point_weight, groups=groups)
            output = contribution if output is None else output + contribution
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)
    return output


def deform_conv2d_compatible(
    input: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Use native torchvision except for the incompatible MUSA kernel.

    JASNA_MUSA_DCN_BACKEND=native can be used to try a newer fixed MUSA wheel.
    """
    backend = os.environ.get("JASNA_MUSA_DCN_BACKEND", "fallback").lower()
    if input.device.type == "musa" and backend != "native":
        return deform_conv2d_grid_sample(input, offset, weight, bias,
                                         stride, padding, dilation, mask)
    return torchvision.ops.deform_conv2d(input, offset, weight, bias,
                                         stride, padding, dilation, mask)
