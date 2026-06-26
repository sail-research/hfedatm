import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SketchConfig:
    mode: str = "diag"
    max_batches: int = 1
    max_patches: int = 2048
    max_full_dim: int = 4096
    block_size: int = 512
    dtype: str = "float32"
    device: str = "cpu"
    shrinkage_alpha: float = 0.75
    dp_epsilon: float = -1.0
    dp_delta: float = 1e-5
    dp_clip: float = 0.0
    lowrank_rank: int = 64
    random_projection_dim: int = 256
    random_seed: int = 0


def _stable_seed(base_seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{int(base_seed)}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2 ** 31)


def _make_projection(dim: int, out_dim: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    projection = torch.randn((int(dim), int(out_dim)), generator=generator, dtype=dtype)
    return projection / math.sqrt(max(int(out_dim), 1))


def materialize_lowrank_gram(C: torch.Tensor, W: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    C = C.detach().cpu().float()
    W = W.detach().cpu().float()
    eye = torch.eye(W.shape[0], dtype=W.dtype)
    middle = torch.linalg.pinv(W + float(ridge) * eye)
    G = C.matmul(middle).matmul(C.T)
    return 0.5 * (G + G.T)


def materialize_random_projection_gram(projection: torch.Tensor, G_projected: torch.Tensor) -> torch.Tensor:
    P = projection.detach().cpu().float()
    Gp = G_projected.detach().cpu().float()
    G = P.matmul(Gp).matmul(P.T)
    return 0.5 * (G + G.T)


class ActivationSketch:
    def __init__(
        self,
        name: str,
        kind: str,
        dim: int,
        mode: str,
        dtype=torch.float32,
        block_size: int = 512,
        lowrank_rank: int = 64,
        random_projection_dim: int = 256,
        random_seed: int = 0,
        max_full_dim: int = 4096,
    ):
        self.name = name
        self.kind = kind
        self.dim = int(dim)
        self.mode = mode
        self.block_size = int(block_size)
        self.max_full_dim = int(max_full_dim)
        self.dtype = dtype
        self.count = 0
        if mode == "diag":
            self.G = torch.zeros(self.dim, dtype=dtype)
        elif mode in {"full", "blockdiag"}:
            self.G = torch.zeros((self.dim, self.dim), dtype=dtype)
        elif mode == "lowrank":
            self.rank = max(1, min(int(lowrank_rank), self.dim))
            self.projection = _make_projection(
                self.dim,
                self.rank,
                dtype=dtype,
                seed=_stable_seed(random_seed, name),
            )
            self.C = torch.zeros((self.dim, self.rank), dtype=dtype)
            self.W = torch.zeros((self.rank, self.rank), dtype=dtype)
            self.G = None
        elif mode == "random_projection":
            self.projection_dim = max(1, min(int(random_projection_dim), self.dim))
            self.projection = _make_projection(
                self.dim,
                self.projection_dim,
                dtype=dtype,
                seed=_stable_seed(random_seed, name),
            )
            self.G_projected = torch.zeros((self.projection_dim, self.projection_dim), dtype=dtype)
            self.G = None
        else:
            raise NotImplementedError(f"Activation sketch mode '{mode}' is not implemented")

    def update(self, X: torch.Tensor):
        X = X.detach()
        if X.numel() == 0:
            return
        X = X.reshape(-1, self.dim).cpu().to(dtype=self.dtype)
        self.count += int(X.shape[0])
        if self.mode == "diag":
            self.G += (X * X).sum(dim=0)
        elif self.mode == "full":
            self.G += X.T.matmul(X)
        elif self.mode == "blockdiag":
            block_size = min(self.dim, max(1, self.block_size))
            for start in range(0, self.dim, block_size):
                end = min(start + block_size, self.dim)
                block = X[:, start:end]
                self.G[start:end, start:end] += block.T.matmul(block)
        elif self.mode == "lowrank":
            Z = X.matmul(self.projection)
            self.C += X.T.matmul(Z)
            self.W += Z.T.matmul(Z)
        elif self.mode == "random_projection":
            Z = X.matmul(self.projection)
            self.G_projected += Z.T.matmul(Z)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "count": int(self.count),
            "mode": self.mode,
            "kind": self.kind,
            "dim": self.dim,
            "max_full_dim": self.max_full_dim,
        }
        if self.mode in {"diag", "full", "blockdiag"}:
            payload["G"] = self.G.detach().cpu()
        elif self.mode == "lowrank":
            payload.update({
                "C": self.C.detach().cpu(),
                "W": self.W.detach().cpu(),
                "projection": self.projection.detach().cpu(),
                "rank": int(self.rank),
            })
            if self.dim <= self.max_full_dim:
                payload["G"] = materialize_lowrank_gram(self.C, self.W).to(dtype=self.C.dtype)
        elif self.mode == "random_projection":
            payload.update({
                "G_projected": self.G_projected.detach().cpu(),
                "projection": self.projection.detach().cpu(),
                "projection_dim": int(self.projection_dim),
            })
            if self.dim <= self.max_full_dim:
                payload["G"] = materialize_random_projection_gram(
                    self.projection,
                    self.G_projected,
                ).to(dtype=self.G_projected.dtype)
        return payload


class ActivationSketcher:
    def __init__(self, model: nn.Module, config: SketchConfig, layer_names: Optional[Iterable[str]] = None):
        self.model = model
        self.config = config
        self.layer_names = set(layer_names) if layer_names is not None else None
        self.handles = []
        self.sketches: Dict[str, ActivationSketch] = {}

    @property
    def root_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _dtype(self):
        return torch.float64 if self.config.dtype == "float64" else torch.float32

    def _mode_for_dim(self, dim: int) -> str:
        mode = self.config.mode
        if mode == "full" and dim > self.config.max_full_dim:
            return "diag"
        return mode

    def _get_or_create(self, name: str, kind: str, dim: int) -> ActivationSketch:
        if name not in self.sketches:
            self.sketches[name] = ActivationSketch(
                name=name,
                kind=kind,
                dim=dim,
                mode=self._mode_for_dim(dim),
                dtype=self._dtype(),
                block_size=self.config.block_size,
                lowrank_rank=self.config.lowrank_rank,
                random_projection_dim=self.config.random_projection_dim,
                random_seed=self.config.random_seed,
                max_full_dim=self.config.max_full_dim,
            )
        return self.sketches[name]

    def _linear_hook(self, name: str):
        def hook(module: nn.Linear, inputs):
            if not inputs:
                return
            X = inputs[0].detach()
            if X.dim() > 2:
                X = X.reshape(-1, X.shape[-1])
            else:
                X = X.reshape(X.shape[0], -1)
            sketch = self._get_or_create(name, "linear", X.shape[-1])
            sketch.update(X)

        return hook

    def _conv_hook(self, name: str):
        def hook(module: nn.Conv2d, inputs):
            if not inputs:
                return
            x = inputs[0].detach()
            unfolded = F.unfold(
                x,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            X = unfolded.transpose(1, 2).reshape(-1, unfolded.shape[1])
            max_patches = int(self.config.max_patches)
            if max_patches > 0 and X.shape[0] > max_patches:
                idx = torch.randperm(X.shape[0], device=X.device)[:max_patches]
                X = X.index_select(0, idx)
            sketch = self._get_or_create(name, "conv2d", X.shape[-1])
            sketch.update(X)

        return hook

    def __enter__(self):
        for name, module in self.root_model.named_modules():
            if self.layer_names is not None and name not in self.layer_names:
                continue
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_pre_hook(self._linear_hook(name)))
            elif isinstance(module, nn.Conv2d):
                self.handles.append(module.register_forward_pre_hook(self._conv_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def payload(self) -> Dict[str, Dict[str, Any]]:
        return {name: sketch.to_payload() for name, sketch in self.sketches.items()}


def aggregate_sketches(client_sketches: List[Dict[str, Dict[str, Any]]], weights: List[float]):
    out: Dict[str, Dict[str, Any]] = {}
    for sketches, weight in zip(client_sketches, weights):
        for name, sketch in sketches.items():
            if name not in out:
                out[name] = {
                    "count": int(sketch.get("count", 0)),
                    "mode": sketch.get("mode", "unknown"),
                    "kind": sketch.get("kind", "unknown"),
                    "dim": int(sketch.get("dim", 0)),
                    "max_full_dim": int(sketch.get("max_full_dim", 4096)),
                }
                for key in ["G", "C", "W", "G_projected"]:
                    if key in sketch and torch.is_tensor(sketch[key]):
                        out[name][key] = float(weight) * sketch[key].detach().cpu()
                if "projection" in sketch and torch.is_tensor(sketch["projection"]):
                    out[name]["projection"] = sketch["projection"].detach().cpu()
                for meta_key in ["rank", "projection_dim"]:
                    if meta_key in sketch:
                        out[name][meta_key] = int(sketch[meta_key])
            else:
                out[name]["count"] += int(sketch.get("count", 0))
                for key in ["G", "C", "W", "G_projected"]:
                    if key in sketch and torch.is_tensor(sketch[key]):
                        value = sketch[key].detach().cpu()
                        if key in out[name] and out[name][key].shape == value.shape:
                            out[name][key] += float(weight) * value
                        else:
                            out[name]["shape_mismatch"] = True
                if "projection" in sketch and "projection" in out[name]:
                    projection = sketch["projection"].detach().cpu()
                    if out[name]["projection"].shape != projection.shape or not torch.allclose(out[name]["projection"], projection):
                        out[name]["projection_mismatch"] = True
    for name, sketch in out.items():
        mode = sketch.get("mode")
        dim = int(sketch.get("dim", 0))
        max_full_dim = int(sketch.get("max_full_dim", 4096))
        if "G" not in sketch and dim <= max_full_dim:
            if mode == "lowrank" and "C" in sketch and "W" in sketch:
                sketch["G"] = materialize_lowrank_gram(sketch["C"], sketch["W"])
            elif mode == "random_projection" and "projection" in sketch and "G_projected" in sketch:
                sketch["G"] = materialize_random_projection_gram(
                    sketch["projection"],
                    sketch["G_projected"],
                )
    return out


def shrink_gram(G: torch.Tensor, alpha: float):
    if G.dim() == 1:
        return G
    diag = torch.diag(torch.diag(G))
    return float(alpha) * G + (1.0 - float(alpha)) * diag


def add_dp_noise_to_gram(G: torch.Tensor, epsilon: float, delta: float, clip: float):
    if epsilon is None or float(epsilon) <= 0 or float(clip) <= 0:
        return G
    sigma = float(clip) * math.sqrt(2.0 * math.log(1.25 / max(float(delta), 1e-12))) / float(epsilon)
    noise = torch.randn_like(G) * sigma
    return G + noise
