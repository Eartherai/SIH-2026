"""Device selection for Apple Silicon (MPS) with an honest CPU fallback.

The project's first-class target is Apple Silicon. We never assume CUDA, but we
do not refuse it either if a user happens to run on an NVIDIA box.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DeviceInfo:
    device: str          # "mps" | "cpu" | "cuda"
    reason: str          # why this device was chosen
    mps_built: bool
    mps_available: bool
    cuda_available: bool
    machine: str
    platform: str

    def as_dict(self) -> dict:
        return asdict(self)


def _probe() -> tuple[bool, bool, bool]:
    try:
        import torch
    except ImportError:
        return False, False, False
    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_avail = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda = bool(torch.cuda.is_available())
    return mps_built, mps_avail, cuda


def select_device(preference: str | None = None) -> DeviceInfo:
    """Resolve the compute device.

    preference: "auto" | "mps" | "cpu" | "cuda" | None (falls back to $AQS_DEVICE, then "auto")
    """
    pref = (preference or os.environ.get("AQS_DEVICE") or "auto").lower()
    mps_built, mps_avail, cuda_avail = _probe()

    common = dict(
        mps_built=mps_built,
        mps_available=mps_avail,
        cuda_available=cuda_avail,
        machine=platform.machine(),
        platform=platform.platform(),
    )

    if pref == "cpu":
        return DeviceInfo("cpu", "explicitly requested", **common)
    if pref == "mps":
        if mps_avail:
            return DeviceInfo("mps", "explicitly requested and available", **common)
        return DeviceInfo("cpu", "mps requested but unavailable; fell back to cpu", **common)
    if pref == "cuda":
        if cuda_avail:
            return DeviceInfo("cuda", "explicitly requested and available", **common)
        return DeviceInfo("cpu", "cuda requested but unavailable; fell back to cpu", **common)

    # auto
    if mps_avail:
        return DeviceInfo("mps", "apple silicon mps detected", **common)
    if cuda_avail:
        return DeviceInfo("cuda", "cuda detected", **common)
    return DeviceInfo("cpu", "no accelerator detected", **common)


def environment_report() -> str:
    """Human-readable environment report (used by run_demo.sh and docs)."""
    import sys

    info = select_device("auto")
    try:
        import torch
        tv = __import__("torchvision").__version__
        torch_v = torch.__version__
    except ImportError:
        torch_v, tv = "not installed", "not installed"

    lines = [
        "Environment",
        "-----------",
        f"OS:            {platform.platform()}",
        f"Architecture:  {platform.machine()}",
        f"Python:        {sys.version.split()[0]}",
        f"PyTorch:       {torch_v}",
        f"torchvision:   {tv}",
        f"MPS built:     {info.mps_built}",
        f"MPS available: {info.mps_available}",
        f"CUDA:          {info.cuda_available}",
        f"Selected:      {info.device}  ({info.reason})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(environment_report())
