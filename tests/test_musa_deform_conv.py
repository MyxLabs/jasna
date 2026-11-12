# SPDX-License-Identifier: AGPL-3.0
"""CPU correctness checks against torchvision's native deform_conv2d."""

import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from jasna.models.basicvsrpp.musa_deform_conv import (
    deform_conv2d_compatible, deform_conv2d_grid_sample,
)


@pytest.mark.parametrize("n,cin,cout,h,w,kh,kw,stride,pad,dil,groups,offset_groups", [
    (1, 3, 5, 10, 10, 3, 3, 1, 0, 1, 1, 1),
    (2, 4, 6, 7, 6, 3, 3, 2, 1, 1, 2, 2),
    (1, 16, 16, 8, 7, 3, 3, 1, 1, 1, 1, 16),
    (1, 2, 4, 5, 6, 2, 2, 1, 0, 2, 1, 1),
    (1, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1),
])
@pytest.mark.parametrize("with_mask", [True, False])
def test_matches_native_cpu(n, cin, cout, h, w, kh, kw, stride, pad, dil,
                            groups, offset_groups, with_mask):
    torch.manual_seed(3)
    oh = (h + 2 * pad - dil * (kh - 1) - 1) // stride + 1
    ow = (w + 2 * pad - dil * (kw - 1) - 1) // stride + 1
    x = torch.randn(n, cin, h, w)
    weight = torch.randn(cout, cin // groups, kh, kw)
    offset = torch.randn(n, 2 * offset_groups * kh * kw, oh, ow) * 0.5
    mask = torch.rand(n, offset_groups * kh * kw, oh, ow) if with_mask else None
    bias = torch.randn(cout)
    native = deform_conv2d(x, offset, weight, bias, stride, pad, dil, mask)
    fallback = deform_conv2d_grid_sample(x, offset, weight, bias, stride, pad, dil, mask)
    torch.testing.assert_close(fallback, native, rtol=5e-4, atol=1e-4)


def test_zero_offsets_matches_convolution():
    torch.manual_seed(123)
    x = torch.randn(2, 8, 8, 7)
    w = torch.randn(6, 4, 3, 3)
    bias = torch.randn(6)
    zeros = torch.zeros(2, 2 * 2 * 9, 8, 7)
    mask = torch.ones(2, 2 * 9, 8, 7)
    out = deform_conv2d_grid_sample(x, zeros, w, bias, 1, 1, 1, mask)
    golden = F.conv2d(x, w, bias, padding=1, groups=2)
    torch.testing.assert_close(out, golden, rtol=5e-5, atol=1e-5)


def test_cpu_keeps_native_path(monkeypatch):
    torch.manual_seed(4)
    x = torch.randn(1, 3, 10, 10)
    w = torch.randn(5, 3, 3, 3)
    offset = torch.rand(1, 18, 8, 8)
    monkeypatch.setenv("JASNA_MUSA_DCN_BACKEND", "fallback")
    torch.testing.assert_close(deform_conv2d_compatible(x, offset, w),
                               deform_conv2d(x, offset, w))


def test_cpu_gradients():
    torch.manual_seed(3)
    x = torch.randn(1, 2, 5, 5, requires_grad=True)
    weight = torch.randn(3, 2, 3, 3, requires_grad=True)
    offset = (torch.randn(1, 18, 3, 3) * 0.15).requires_grad_()
    mask = torch.rand(1, 9, 3, 3, requires_grad=True)
    bias = torch.randn(3, requires_grad=True)
    native = deform_conv2d(x, offset, weight, bias, mask=mask)
    grad = torch.randn_like(native)
    native.backward(grad)
    grads_native = [t.grad.clone() for t in (x, offset, weight, mask, bias)]
    for t in (x, offset, weight, mask, bias):
        t.grad = None
    fallback = deform_conv2d_grid_sample(x, offset, weight, bias, mask=mask)
    fallback.backward(grad)
    for t, grad_native in zip((x, offset, weight, mask, bias), grads_native):
        torch.testing.assert_close(t.grad, grad_native, rtol=5e-4, atol=1e-4)
