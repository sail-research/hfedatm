import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets import PACS


def build_cache(args):
    dataset = PACS(
        version=args.version,
        root_dir=args.root_dir,
        download=args.download,
        split_scheme=args.split_scheme,
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else dataset.data_dir / "tensor_cache_224_float32_imagenet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / "images.npy"
    tmp_cache_file = cache_dir / "images.tmp.npy"
    metadata_file = cache_dir / "metadata.json"

    if cache_file.exists() and not args.force:
        print(f"Cache already exists: {cache_file}")
        print("Use --force to rebuild it.")
        return

    if tmp_cache_file.exists():
        tmp_cache_file.unlink()

    image_size = int(args.image_size)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    start = time.time()
    shape = (len(dataset), 3, image_size, image_size)
    cache = np.lib.format.open_memmap(
        tmp_cache_file,
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )

    for idx in tqdm(range(len(dataset)), desc="Caching PACS tensors"):
        image = dataset.get_input(idx)
        tensor = transform(image).to(dtype=torch.float32)
        cache[idx] = tensor.numpy()

    cache.flush()
    del cache
    tmp_cache_file.replace(cache_file)

    metadata = {
        "dataset": "PACS",
        "version": args.version,
        "split_scheme": args.split_scheme,
        "num_examples": len(dataset),
        "shape": list(shape),
        "dtype": "float32",
        "image_size": image_size,
        "preprocessing": "Resize(224,224) -> ToTensor -> ImageNet Normalize",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "source_data_dir": str(dataset.data_dir),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote tensor cache: {cache_file}")
    print(f"Wrote metadata: {metadata_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a pre-resized, pre-normalized PACS tensor cache.")
    parser.add_argument("--root_dir", default="data", help="Dataset root containing pacs_v1.0.")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--split_scheme", default="official")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--image_size", default=224, type=int)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
