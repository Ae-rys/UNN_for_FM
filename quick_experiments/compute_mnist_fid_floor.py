# -*- coding: utf-8 -*-
"""
compute_mnist_fid_floor.py
Le PLANCHER du mini-FID MNIST, au MEME protocole que les FID des modeles.

Pourquoi : make_mnist_report_figs.py compare 2000 images generees a une reference
de 2000 images d'entrainement (les 2000 premieres du dataset). Le plancher annonce
dans le rapport (1.03) a lui ete mesure sur 10k/10k — donc a un N cinq fois plus
grand, ou le biais du FID est bien moindre. Les deux chiffres ne sont pas sur la
meme echelle et ne doivent pas figurer dans la meme colonne.

On mesure ici, avec R = les 2000 memes images de reference :
  - R vs 2000 images de TEST          -> vrai train/test, le plancher a citer
  - R vs 2000 AUTRES images de train  -> isole le bruit d'echantillonnage du
                                         decalage train/test
  - la dependance en N, parce que le FID decroit fortement avec N (cf. AFHQ).

Usage :  python compute_mnist_fid_floor.py [--n-eval 2000]
"""
import argparse
import torch
import torchvision
from torchvision import transforms

from mnist_metrics import train_or_load_classifier, mini_fid
from run_mnist import get_train_loader

IMG = 28


def as_images(ds, idx):
    return torch.stack([ds[i][0] for i in idx]).view(-1, 1, IMG, IMG).clamp(-1, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    dev = torch.device(a.device)
    N = a.n_eval

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train = torchvision.datasets.MNIST(root="./data", train=True, download=False, transform=tf)
    test = torchvision.datasets.MNIST(root="./data", train=False, download=False, transform=tf)
    clf = train_or_load_classifier(get_train_loader(batch_size=256, digit=None), dev)

    # R : EXACTEMENT la reference de make_mnist_report_figs (les N premieres de train)
    R = as_images(train, range(N))
    test_set = as_images(test, range(N))              # vrai jeu de test
    train2 = as_images(train, range(N, 2 * N))        # autres images de train, disjointes

    print(f"\nreference R = {N} premieres images de TRAIN (identique au depouillement)\n")
    print(f"{'Jeu compare a R':<42}{'N':>7}{'mini-FID':>11}")
    print("-" * 60)
    f_test = mini_fid(clf, test_set, R, dev)
    f_tr2 = mini_fid(clf, train2, R, dev)
    print(f"{'2000 images de TEST (plancher train/test)':<42}{N:>7}{f_test:>11.2f}")
    print(f"{'2000 AUTRES images de TRAIN':<42}{N:>7}{f_tr2:>11.2f}")

    print(f"\nDependance en N (test vs R, tailles egales) :")
    print(f"{'N':>7}{'mini-FID':>11}")
    for n in [250, 500, 1000, 2000, 5000]:
        if n > len(test): break
        r = as_images(train, range(n))
        t = as_images(test, range(n))
        print(f"{n:>7}{mini_fid(clf, t, r, dev):>11.2f}")

    print(f"\nA comparer aux modeles a N={N} : SmallUNet 27.6 | ScCP LNO 41.4 | "
          f"ScCP LFO 53.3 | DFB LFO 121.0 | DFB LNO 275.2")


if __name__ == "__main__":
    main()
