# SPDX-License-Identifier: AGPL-3.0
"""MTT S4000/MUSA BasicVSR++ runtime smoke test.

Run in AutoDL GPU-attached mode:
    /usr/bin/python3 scripts/check_musa_basicvsrpp.py --checkpoint model_weights/lada_mosaic_restoration_model_generic_v1.2.pth
"""
import argparse
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()
    import torch
    import torch_musa  # noqa: F401 - register MUSA backend
    from jasna.models.basicvsrpp.inference import load_model
    from jasna.models.basicvsrpp.musa_deform_conv import deform_conv2d_grid_sample

    print("Python:", sys.version.split()[0], "Torch:", torch.__version__, flush=True)
    if not torch.musa.is_available():
        print("MUSA unavailable; attach the MTT S4000 before running this test.", flush=True)
        return 2
    print("GPU:", torch.musa.get_device_name(0), flush=True)

    a = torch.randn(128, 128, device="musa")
    torch.musa.synchronize()
    print("MUSA matmul:", (a @ a).shape, flush=True)

    x = torch.randn(1, 3, 10, 10, device="musa")
    w = torch.randn(5, 3, 3, 3, device="musa")
    offset = torch.zeros(1, 18, 8, 8, device="musa")
    mask = torch.ones(1, 9, 8, 8, device="musa")
    actual = deform_conv2d_grid_sample(x, offset, w, mask=mask)
    expected = torch.nn.functional.conv2d(x, w)
    torch.musa.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=2e-4, rtol=2e-4)
    print("MUSA DCNv2 fallback (1 offset group): PASS", flush=True)

    x = torch.randn(1, 128, 16, 16, device="musa")
    w = torch.randn(64, 128, 3, 3, device="musa") * 0.03
    offset = torch.zeros(1, 16 * 18, 16, 16, device="musa")
    mask = torch.ones(1, 16 * 9, 16, 16, device="musa")
    actual = deform_conv2d_grid_sample(x, offset, w, padding=1, mask=mask)
    expected = torch.nn.functional.conv2d(x, w, padding=1)
    torch.musa.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=5e-4, rtol=5e-4)
    print("MUSA DCNv2 fallback (16 offset groups): PASS", flush=True)

    print("Loading:", args.checkpoint, flush=True)
    model = load_model(None, args.checkpoint, torch.device("musa"), False)
    print("BasicVSR++ checkpoint and MUSA model loading: PASS", flush=True)
    x = torch.rand(1, args.frames, 3, args.height, args.width, device="musa")
    t0 = time.perf_counter()
    with torch.inference_mode():
        result = model(inputs=x, mode="tensor")
    torch.musa.synchronize()
    if not isinstance(result, torch.Tensor):
        raise TypeError("Unexpected forward result type: " + str(type(result)))
    if not torch.isfinite(result).all().item():
        raise ValueError("BasicVSR++ produced NaN or Inf")
    print("BasicVSR++ MUSA forward: PASS; output:", tuple(result.shape),
          "seconds:", round(time.perf_counter()-t0, 2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
