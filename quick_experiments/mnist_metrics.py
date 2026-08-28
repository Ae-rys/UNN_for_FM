# -*- coding: utf-8 -*-
"""
mnist_metrics.py
Metriques de qualite reutilisables pour des echantillons MNIST generes,
dans l'esprit de compute_fid_cifar10.py (clean-fid + Inception) mais adaptees
a MNIST (pas de stats Inception officielles pour des images 28x28 en niveaux
de gris) :

  - mini-FID  : distance de Frechet dans l'espace des features d'un petit
                classifieur CNN entraine sur MNIST (au lieu d'Inception).
                Pas comparable a un FID publie ailleurs, mais un CLASSEMENT
                relatif valide entre nos propres configs (meme protocole).
  - entropie de classe : couverture des 10 chiffres (1.0 = uniforme, 0 = collapse
                total sur un seul chiffre) via le meme classifieur.
  - std_gen   : diversite brute pixel (detecteur de collapse bon marche, deja
                utilise dans grille_convsccp.py / track_latent_extension).
  - distance-to-train : ratio distance-au-train des echantillons generes vs
                distance-au-train d'un vrai lot de test (>1 = generalise,
                <=1 = suspect de memorisation), cf. overfit_test_convsccp.py.

Le classifieur est entraine une fois et mis en cache sur disque
(results/mnist_classifier/clf.pt) pour ne pas repayer son cout a chaque config
du sweep.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg


class MNISTClassifier(nn.Module):
    """Petit CNN 10 classes. `features()` expose la couche penultieme (128-d)
    utilisee comme embedding pour le mini-FID."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
        )
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x):
        h = self.conv(x).flatten(1)
        return F.relu(self.fc1(h))

    def forward(self, x):
        return self.fc2(self.features(x))


def train_or_load_classifier(train_loader, device, ckpt_path="results/mnist_classifier/clf.pt",
                              epochs=3):
    """Entraine (ou recharge) le classifieur 10-classes sur `train_loader`
    (images normalisees [-1,1], comme le reste du pipeline FM). ~3 epoques
    suffisent pour >98% d'accuracy sur MNIST, quelques dizaines de secondes."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    clf = MNISTClassifier().to(device)
    if os.path.exists(ckpt_path):
        clf.load_state_dict(torch.load(ckpt_path, map_location=device))
        clf.eval()
        print(f"[mnist_metrics] classifieur charge depuis {ckpt_path}", flush=True)
        return clf

    print(f"[mnist_metrics] entrainement classifieur ({epochs} epoques)...", flush=True)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    clf.train()
    for ep in range(epochs):
        n_correct, n_total, tot_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item()
            n_correct += (logits.argmax(1) == y).sum().item()
            n_total += y.size(0)
        print(f"  [clf] epoch {ep+1}/{epochs} loss={tot_loss/len(train_loader):.4f} "
              f"acc={n_correct/n_total:.4f}", flush=True)
    clf.eval()
    torch.save(clf.state_dict(), ckpt_path)
    print(f"[mnist_metrics] classifieur sauve -> {ckpt_path}", flush=True)
    return clf


@torch.no_grad()
def _embed_all(clf, images, device, batch_size=512):
    """images: (N,1,28,28) sur CPU ou GPU, deja dans la meme normalisation
    ([-1,1]) que l'entrainement du classifieur. Retourne (N,128) numpy."""
    feats = []
    for i in range(0, images.shape[0], batch_size):
        batch = images[i:i + batch_size].to(device)
        feats.append(clf.features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Formule FID standard (Heusel et al.), sur des stats deja calculees."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


@torch.no_grad()
def mini_fid(clf, gen_images, real_images, device):
    """mini-FID = distance de Frechet dans l'espace de features du classifieur,
    entre generes et reels. Comparable UNIQUEMENT entre runs de CE script
    (pas un FID publie, pas de stats de reference externes)."""
    feat_g = _embed_all(clf, gen_images, device)
    feat_r = _embed_all(clf, real_images, device)
    mu_g, sigma_g = feat_g.mean(0), np.cov(feat_g, rowvar=False)
    mu_r, sigma_r = feat_r.mean(0), np.cov(feat_r, rowvar=False)
    return frechet_distance(mu_g, sigma_g, mu_r, sigma_r)


@torch.no_grad()
def class_entropy(clf, gen_images, device, n_classes=10):
    """Entropie normalisee (0-1) de la distribution des classes predites sur
    les echantillons generes. 1.0 = couverture uniforme des 10 chiffres,
    0.0 = collapse total sur une seule classe."""
    logits = []
    for i in range(0, gen_images.shape[0], 512):
        logits.append(clf(gen_images[i:i + 512].to(device)))
    preds = torch.cat(logits, dim=0).argmax(1).cpu().numpy()
    counts = np.bincount(preds, minlength=n_classes).astype(np.float64)
    p = counts / counts.sum()
    p_nz = p[p > 0]
    h = -(p_nz * np.log(p_nz)).sum()
    return float(h / np.log(n_classes)), counts.astype(int).tolist()


@torch.no_grad()
def distance_to_train_ratio(gen_images, train_pool, test_pool, pool_size=3000):
    """Ratio (dist generes->train) / (dist test_reel->train). ~1 = le modele
    genere des echantillons aussi 'nouveaux' que de vraies images non-vues.
    << 1 = suspect de memorisation. Sous-echantillonne le pool pour rester
    rapide (kNN brut en pixels, cf. overfit_test_convsccp.py)."""
    train_pool = train_pool[:pool_size].flatten(1)
    gen_flat = gen_images.flatten(1)
    test_flat = test_pool.flatten(1)

    def mean_nn_dist(query, pool):
        d = torch.cdist(query, pool)          # (Nq, Npool)
        return d.min(dim=1).values.mean().item()

    dist_gen = mean_nn_dist(gen_flat, train_pool)
    dist_test = mean_nn_dist(test_flat, train_pool)
    return dist_gen / dist_test, dist_gen, dist_test
