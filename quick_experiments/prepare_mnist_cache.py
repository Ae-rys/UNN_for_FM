# -*- coding: utf-8 -*-
"""
prepare_mnist_cache.py

Cache MNIST au MEME format que ./data/afhq_cat32_train.pt, pour que
denoise_probe.py tourne dessus sans modification :

    {"data": uint8 (N, 1, 28, 28), "labels": int64 (N,)}

denoise_probe deduit C et S de la forme du cache et gere deja le niveau de gris
a l'affichage, donc rien d'autre a toucher.

MNIST est deja telecharge dans ./data/MNIST — aucun acces reseau necessaire.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python prepare_mnist_cache.py            # -> ./data/mnist_train.pt
"""

import argparse
import os
import time

import torch
from torchvision import datasets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data")
    p.add_argument("--out", type=str, default="./data/mnist_train.pt")
    args = p.parse_args()

    t0 = time.perf_counter()
    ds = datasets.MNIST(root=args.root, train=True, download=False)
    x = ds.data.unsqueeze(1).contiguous()              # (N,28,28) -> (N,1,28,28) uint8
    y = ds.targets.to(torch.long)
    assert x.dtype == torch.uint8 and x.shape[1:] == (1, 28, 28), x.shape
    torch.save({"data": x, "labels": y}, args.out)
    print(f"Cache -> {args.out} : {tuple(x.shape)} uint8, "
          f"{os.path.getsize(args.out)/1e6:.0f} Mo, {time.perf_counter()-t0:.1f}s",
          flush=True)
    print(f"  min={int(x.min())} max={int(x.max())}  "
          f"-> apres normalisation [-1,1] : var={float((x.float()/127.5-1).var()):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
