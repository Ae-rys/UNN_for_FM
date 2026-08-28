# -*- coding: utf-8 -*-
"""
prepare_afhq_cats.py
Construit le cache AFHQ-chats 32x32 pour run_afhq32.py.

Source : dataset HuggingFace `huggan/AFHQ` (AFHQ v1, Choi et al. 2020),
16 130 images 512x512 (cat/dog/wild) dans 2 fichiers parquet (~730 Mo).
On telecharge les parquets, on filtre label==cat (~5 150 images), on
redimensionne en 32x32 (LANCZOS) et on sauve un cache au MEME format que
./data/cifar10_train_rgb.pt : {"data": uint8 (N,3,32,32), "labels": long (N,)}.

Les parquets bruts sont gardes dans ./data/afhq_parquet/ (re-execution ou
future variante 64x64 sans re-telecharger). Le script est idempotent :
fichiers deja presents -> pas re-telecharges ; cache deja present -> stop.

Usage
-----
    python prepare_afhq_cats.py                 # -> ./data/afhq_cat32_train.pt
    python prepare_afhq_cats.py --size 64       # -> ./data/afhq_cat64_train.pt
"""

import argparse
import io
import os
import time
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

PARQUET_URLS = [
    "https://huggingface.co/api/datasets/huggan/AFHQ/parquet/default/train/0.parquet",
    "https://huggingface.co/api/datasets/huggan/AFHQ/parquet/default/train/1.parquet",
]
LABEL_CAT = 0          # ClassLabel names = ["cat", "dog", "wild"]


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  deja la : {dest} ({os.path.getsize(dest)/1e6:.0f} Mo)", flush=True)
        return
    print(f"  telechargement {url}", flush=True)
    t0 = time.perf_counter()

    def hook(nblocks, bs, total):
        done = nblocks * bs
        if nblocks % 2000 == 0 and total > 0:
            mbps = done / 1e6 / max(time.perf_counter() - t0, 1e-9)
            print(f"    {done/1e6:6.0f}/{total/1e6:.0f} Mo  ({mbps:.1f} Mo/s)", flush=True)

    tmp = dest + ".tmp"
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    os.replace(tmp, dest)
    print(f"  ok -> {dest} ({os.path.getsize(dest)/1e6:.0f} Mo, "
          f"{time.perf_counter()-t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Cache AFHQ-chats redimensionne.")
    p.add_argument("--size", type=int, default=32, help="Cote cible (32 ou 64).")
    p.add_argument("--data-dir", type=str, default="./data")
    args = p.parse_args()

    cache = os.path.join(args.data_dir, f"afhq_cat{args.size}_train.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        print(f"Cache deja present : {cache} ({d['data'].shape[0]} images). Rien a faire.")
        return

    pq_dir = os.path.join(args.data_dir, "afhq_parquet")
    os.makedirs(pq_dir, exist_ok=True)
    files = []
    for url in PARQUET_URLS:
        dest = os.path.join(pq_dir, url.rsplit("/", 1)[-1])
        download(url, dest)
        files.append(dest)

    imgs = []
    t0 = time.perf_counter()
    for f in files:
        tbl = pq.read_table(f, columns=["image", "label"])
        labels = np.asarray(tbl["label"])
        rows = tbl["image"].to_pylist()          # dicts {"bytes": ..., "path": ...}
        n_cat = int((labels == LABEL_CAT).sum())
        print(f"  {os.path.basename(f)} : {len(rows)} images, {n_cat} chats", flush=True)
        for i, (row, lab) in enumerate(zip(rows, labels)):
            if lab != LABEL_CAT:
                continue
            im = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
            im = im.resize((args.size, args.size), Image.LANCZOS)
            imgs.append(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
        del tbl, rows

    x = torch.from_numpy(np.stack(imgs))                       # (N,3,s,s) uint8
    torch.save({"data": x, "labels": torch.zeros(x.shape[0], dtype=torch.long)}, cache)
    print(f"\nCache -> {cache} : {tuple(x.shape)} uint8, "
          f"{os.path.getsize(cache)/1e6:.0f} Mo, {time.perf_counter()-t0:.0f}s", flush=True)

    # apercu visuel pour verifier le contenu (grille 8x8)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        for k, ax in enumerate(axes.flat):
            ax.imshow(x[k].permute(1, 2, 0).numpy())
            ax.axis("off")
        fig.suptitle(f"AFHQ chats {args.size}x{args.size} — apercu cache")
        preview = cache.replace(".pt", "_preview.png")
        plt.tight_layout(); plt.savefig(preview, dpi=90); plt.close(fig)
        print(f"Apercu -> {preview}", flush=True)
    except Exception as exc:
        print(f"(apercu saute : {exc})", flush=True)


if __name__ == "__main__":
    main()
