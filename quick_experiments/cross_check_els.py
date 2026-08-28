"""Cross-check : mon ELS est-il fidèle à la VRAIE fonction de Kamb ?

Son LocalScoreModule calcule le score local-équivariant per-pixel (softmax de patchs
zero-pad), convention VP (a=√(1-β), b=√β). On le sous-classe en corrigeant UNIQUEMENT
un bug de broadcast standalone (denominator sans [:,None]) — sa logique de calcul est
intacte. Pont FM→VP (le débruiteur ne dépend que du SNR σ=b/a et du point rescalé) :
  σ=(1-t)/t, β=σ²/(1+σ²), a=√(1-β), b=√β, x_v=a·(xt/t), Tweedie D=(x_v+b²·score)/a=E[x1|xt].
On compare D_kamb à mes ex1_els_center (= sa convention) et ex1_els_nifty (fold NIFTY).
"""
import sys, os, torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import nifty_els_fm as N
sys.path.append(os.path.expanduser("~/repro_kamb"))
from utils.idealscore import LocalScoreModule

dev = N.dev
K, NSUB, NQ = 7, 2000, 8
TS = [0.3, 0.5, 0.7, 0.85]


class LSMfix(LocalScoreModule):
    """LocalScoreModule de Kamb, logique INCHANGÉE, seul le broadcast final corrigé
    (denominator[:,None] au lieu de denominator) pour marcher en standalone c=1."""
    def forward(self, t, x, label=None, device=None, k=None):
        if k is None: k = self.kernel_size
        if device is None: device = dev
        x = x.to(device)
        b, c, h, w = x.shape
        bt = (self.schedule(t)) ** 0.5; at = (1 - self.schedule(t)) ** 0.5
        at = torch.as_tensor(at).to(device); bt = torch.as_tensor(bt).to(device)
        numerator = torch.zeros(x.shape, device=device)
        denominator = torch.zeros(b, h, w, device=device)
        subtraction = None; i = 0
        for images, labels in self.trainloader:
            if label is not None: images = images[(labels == label).squeeze(), :, :, :]
            if images.shape[0] == 0: continue
            images = images.to(device); bsize = images.shape[0]
            i += bsize
            if self.max_samples is not None and i > self.max_samples: break
            pwise_diffs = x[:, None, :, :, :] - at * images[None, :, :, :, :]
            pwise_normsquares = torch.sum(pwise_diffs ** 2, dim=2)
            patches = F.unfold(pwise_normsquares, k, stride=1, padding=k // 2).view(b, bsize, k ** 2, h, w)
            exp_args = -torch.sum(patches, dim=2) / (2 * bt ** 2)
            if subtraction is None:
                subtraction = torch.amax(exp_args, dim=(0, 1), keepdim=True)
            else:
                new_s = torch.amax(exp_args, dim=(0, 1), keepdim=True)
                delta = torch.maximum(new_s, subtraction)
                numerator /= torch.exp(delta - subtraction)
                denominator /= torch.exp(delta - subtraction)[:, 0, :, :]
                subtraction = delta
            exp_vals = torch.exp(exp_args - subtraction)
            numerator += torch.mean(exp_vals[:, :, None, :, :] * pwise_diffs, dim=1)
            denominator += torch.mean(exp_vals, dim=1)
        return -numerator / denominator[:, None] / bt ** 2          # <-- FIX broadcast


def cos_unc(a, b): return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)
def rel_err(a, b): return (a - b).norm(dim=1) / (a.norm(dim=1) + 1e-12)


def main():
    Xsub = N.mnist_train(NSUB, seed=0); X1 = Xsub.view(NSUB, N.DIM)
    pat, pn = N.build_dict(Xsub, K); gw = N.gauss_patch_weight(K)
    cen = X1.reshape(-1).contiguous()
    ds = TensorDataset(Xsub.cpu(), torch.zeros(NSUB, dtype=torch.long))
    lsm = LSMfix(ds, kernel_size=K, image_size=28, batch_size=64,
                 schedule=(lambda tau: tau), max_samples=NSUB)
    Xq = N.mnist_train(NQ, seed=1).view(NQ, N.DIM)

    lines = [f"# cross-check ELS : Kamb LocalScoreModule(fix) vs ex1_els_* | K={K} NSUB={NSUB} NQ={NQ}",
             "t\tcos(kamb,center)\trel(center)\tcos(kamb,nifty)\trel(nifty)\tcos(center,nifty)"]
    for t_fm in TS:
        xt = t_fm * Xq + (1 - t_fm) * torch.randn_like(Xq)
        sigma = (1 - t_fm) / t_fm; beta = sigma ** 2 / (1 + sigma ** 2)
        at = (1 - beta) ** 0.5; bt = beta ** 0.5
        x_v = (at * (xt / t_fm)).view(NQ, 1, N.S, N.S)
        with torch.no_grad():
            score = lsm(torch.tensor(float(beta)), x_v, k=K, device=dev)
            D_kamb = ((x_v + bt ** 2 * score) / at).view(NQ, N.DIM)
            D_center = N.ex1_els_center(xt, t_fm, pat, pn, cen, K)
            D_nifty = N.ex1_els_nifty(xt, t_fm, pat, pn, K, gw)
        cc, rc = cos_unc(D_kamb, D_center).median().item(), rel_err(D_kamb, D_center).median().item()
        cn, rn = cos_unc(D_kamb, D_nifty).median().item(), rel_err(D_kamb, D_nifty).median().item()
        ccn = cos_unc(D_center, D_nifty).median().item()
        line = f"{t_fm:.2f}\t{cc:.4f}\t{rc:.4f}\t{cn:.4f}\t{rn:.4f}\t{ccn:.4f}"
        print(line, flush=True); lines.append(line)
    open("cross_check_els.txt", "w").write("\n".join(lines) + "\n")
    print("saved -> cross_check_els.txt", flush=True)


if __name__ == "__main__":
    main()
