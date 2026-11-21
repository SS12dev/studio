"""Candidate models for still-image extraction, each behind the same tiny interface.

A runner loads once, then maps an RGB uint8 image to a single-channel or RGB uint8 image that
can be saved and compared side by side. Real outputs (float EXR passes) come later; this is for
choosing models, not for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _to_u8(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


@dataclass(frozen=True)
class Runner:
    name: str
    stage: str
    licence: str
    load: Callable[[], Callable[[np.ndarray], np.ndarray]]


# matting ------------------------------------------------------------------
def _birefnet(repo: str):
    def load():
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        dev = device()
        model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
        model = model.to(dev).eval().half() if dev == "mps" else model.eval()
        tf = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        @torch.inference_mode()
        def run(rgb: np.ndarray) -> np.ndarray:
            img = Image.fromarray(rgb)
            x = tf(img).unsqueeze(0).to(dev)
            x = x.half() if dev == "mps" else x
            pred = model(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
            alpha = Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size)
            return np.asarray(alpha)

        return run

    return load


# depth --------------------------------------------------------------------
def _hf_depth(repo: str):
    def load():
        from transformers import pipeline

        pipe = pipeline("depth-estimation", model=repo, device=device())

        def run(rgb: np.ndarray) -> np.ndarray:
            out = pipe(Image.fromarray(rgb))
            return np.asarray(out["depth"].convert("L"))  # normalised visualisation

        return run

    return load


# intrinsics (albedo) --------------------------------------------------------
def _marigold_iid(repo: str):
    def load():
        import torch
        from diffusers import MarigoldIntrinsicsPipeline

        dev = device()
        dtype = torch.float16 if dev == "mps" else torch.float32
        try:
            pipe = MarigoldIntrinsicsPipeline.from_pretrained(repo, variant="fp16", torch_dtype=dtype)
        except Exception:  # noqa: BLE001 - repo may not publish an fp16 variant
            pipe = MarigoldIntrinsicsPipeline.from_pretrained(repo, torch_dtype=dtype)
        pipe = pipe.to(dev)

        def run(rgb: np.ndarray) -> np.ndarray:
            img = Image.fromarray(rgb)
            out = pipe(img, num_inference_steps=4, processing_resolution=768)
            vis = pipe.image_processor.visualize_intrinsics(out.prediction, pipe.target_properties)
            albedo = vis[0]["albedo"]  # first target is albedo for both IID variants
            return np.asarray(albedo.resize(img.size).convert("RGB"))

        return run

    return load


RUNNERS: dict[str, Runner] = {r.name: r for r in [
    Runner("birefnet", "matting", "MIT", _birefnet("ZhengPeng7/BiRefNet")),
    Runner("birefnet-lite", "matting", "MIT", _birefnet("ZhengPeng7/BiRefNet_lite")),
    Runner("depth-anything-v2-small", "depth", "Apache-2.0",
           _hf_depth("depth-anything/Depth-Anything-V2-Small-hf")),
    Runner("depth-anything-v2-base", "depth", "CC-BY-NC-4.0",
           _hf_depth("depth-anything/Depth-Anything-V2-Base-hf")),
    Runner("depth-pro", "depth", "Apple sample-code licence", _hf_depth("apple/DepthPro-hf")),
    Runner("marigold-iid-lighting", "albedo", "see model card",
           _marigold_iid("prs-eth/marigold-iid-lighting-v1-1")),
    Runner("marigold-iid-appearance", "albedo", "see model card",
           _marigold_iid("prs-eth/marigold-iid-appearance-v1-1")),
]}
