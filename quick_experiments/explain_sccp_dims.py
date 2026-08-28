# -*- coding: utf-8 -*-
"""
explain_sccp_dims.py
Trace dimensionnelle d'UNE iteration `_OrigConvScCP_Iteration`, mesuree sur un
vrai passage avant plutot que deduite du code.

Ce qu'on cherche a rendre visible
---------------------------------
  * l'asymetrie primal / dual : le primal reste une image a `in_channels`
    canaux, le dual vit dans un espace `internal_channel` fois plus large ;
  * ou passent les parametres (W, V, biais, prox) et dans quelles proportions ;
  * comment W (conv2d) et V (conv_transpose2d) font l'aller-retour entre les
    deux espaces, et pourquoi ils ont la MEME forme de noyau ;
  * la croissance du champ receptif avec la profondeur K ;
  * ce que le prox voit exactement (et ce qu'il ne voit pas).

Le script compare aussi le cout memoire de l'etat dual a celui de l'image, qui
est la vraie raison pour laquelle un ScCP est lent.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python explain_sccp_dims.py                       # config AFHQ par defaut
    python explain_sccp_dims.py --ic 256 --K 10 --kernel 15

Sortie : un tableau sur la sortie standard (+ --out pour l'ecrire dans un .txt).
"""

import argparse

import torch
import torch.nn.functional as F

from models.architectures import ConvScCP_UNN


def human(n):
    return f"{n:,}".replace(",", " ")


def main():
    p = argparse.ArgumentParser(description="Trace dimensionnelle du bloc ScCP.")
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--ic", type=int, default=128, help="internal_channel = dim duale")
    p.add_argument("--kernel", type=int, default=9)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prox", type=str, default="l1", choices=["l1", "l1c"])
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    C, S, ic, k, K, B = (args.in_channels, args.img_size, args.ic,
                         args.kernel, args.K, args.batch)
    dim = C * S * S
    lines = []

    def say(s=""):
        print(s, flush=True); lines.append(s)

    model = ConvScCP_UNN(dim=dim, K=K, internal_channel=ic, kernel_size=k,
                         in_channels=C, img_size=S, use_Unet="l1",
                         version="LFO", w_bias=True)
    if args.prox == "l1c":
        from models.architectures import L1ProxConv
        for layer in model.layers:
            layer.prox = L1ProxConv(w=32, channels=ic)
    layer = model.layers[0]
    pad = layer.pad

    say("=" * 78)
    say(f"ConvScCP  in_channels={C}  img={S}x{S}  ic={ic}  kernel={k}  K={K}  prox={args.prox}")
    say("=" * 78)

    # ---------------- les deux espaces ----------------
    say("\nLES DEUX ESPACES")
    say(f"  primal  x, z   (B, {C}, {S}, {S})   = {human(dim)} valeurs par image")
    say(f"  dual    u      (B, {ic}, {S}, {S})   = {human(ic*S*S)} valeurs par image")
    say(f"  rapport dual/primal : x{ic*S*S/dim:.1f}   (= internal_channel / in_channels)")
    say(f"  -> le dual est le vrai etat du reseau ; le primal reste une image.")

    # ---------------- parametres ----------------
    say("\nPARAMETRES D'UNE COUCHE")
    tot = sum(q.numel() for q in layer.parameters())
    for name, q in layer.named_parameters():
        say(f"  {name:<28}{str(tuple(q.shape)):<20}{human(q.numel()):>10}"
            f"   {100*q.numel()/tot:5.1f} %")
    say(f"  {'TOTAL par couche':<28}{'':<20}{human(tot):>10}")
    say(f"  {'x K = ' + str(K) + ' couches':<28}{'':<20}"
        f"{human(sum(q.numel() for q in model.parameters())):>10}")

    # ---------------- trace du passage avant ----------------
    say("\nTRACE D'UNE ITERATION (formes reelles, batch = %d)" % B)
    z = torch.randn(B, C, S, S)
    x = z.clone()
    u = torch.zeros(B, ic, S, S)
    t = torch.rand(B, 1)
    tau = F.softplus(model.log_tau0).detach()
    sigma = torch.tensor(1.0)
    alpha = (1.0 + 2.0 * tau).pow(-0.5)
    V = layer.V_weight

    with torch.no_grad():
        back = F.conv_transpose2d(u, V, padding=pad)
        say(f"  1. V*u        conv_transpose2d(u{tuple(u.shape)}, V{tuple(V.shape)}, pad={pad})")
        say(f"     {'':>13}-> {tuple(back.shape)}    dual -> primal")
        primal_input = x - tau * back
        x_next = (primal_input + tau * z) / (1 + tau)
        say(f"  2. x_next     (x - tau*V*u + tau*z) / (1+tau)  -> {tuple(x_next.shape)}")
        say(f"     {'':>13}   tau = {tau.item():.4f} : reinjection de z a CHAQUE iteration")
        y = x_next + alpha * (x_next - x)
        say(f"  3. y          x_next + alpha*(x_next - x)      -> {tuple(y.shape)}")
        say(f"     {'':>13}   alpha = {alpha.item():.4f} (inertie, pas de parametre)")
        Wy = F.conv2d(y, layer.W_weight, bias=layer.W_bias, padding=pad)
        say(f"  4. W*y        conv2d(y{tuple(y.shape)}, W{tuple(layer.W_weight.shape)}, pad={pad})")
        say(f"     {'':>13}-> {tuple(Wy.shape)}   primal -> dual")
        u_next = layer.prox(u + sigma * Wy, t)
        say(f"  5. u_next     prox(u + sigma*W*y, t)           -> {tuple(u_next.shape)}")
        r = F.softplus(layer.prox.time_scaling(t))
        say(f"     {'':>13}   rayon r(t) de forme {tuple(r.shape)} "
            f"= {'UN scalaire par image' if r.shape[1] == 1 else str(r.shape[1]) + ' rayons, un par canal dual'}")

    # ---------------- ce que le prox voit ----------------
    say("\nCE QUE LE PROX VOIT")
    say(f"  entree  : u de forme (B, {ic}, {S}, {S}) = {human(ic*S*S)} valeurs")
    say(f"  parametres du prox : {human(sum(q.numel() for q in layer.prox.parameters()))}"
        f"  (MLP 1 -> 32 -> {r.shape[1]})")
    say(f"  operation : clamp(u, -r(t), +r(t)), PIXEL PAR PIXEL")
    say(f"  -> aucune interaction spatiale, aucune interaction entre canaux.")
    say(f"  -> c'est la SEULE non-linearite du bloc.")

    # ---------------- champ receptif ----------------
    say("\nCHAMP RECEPTIF (theorique, sans les effets de bord)")
    say(f"  un conv {k}x{k} ajoute +/-{pad} pixels ; une iteration en enchaine deux")
    say(f"  (W puis V), soit +/-{2*pad} pixels par iteration.")
    for kk in (1, 2, 5, 10, K):
        rf = 2 * (2 * pad) * kk + 1
        flag = "  <- couvre toute l'image" if rf >= S else ""
        say(f"    apres {kk:>3} iteration(s) : {rf:>4} x {rf:<4} pixels{flag}")
    say(f"  l'image fait {S}x{S} : la localite cesse d'etre une contrainte "
        f"des {int((S-1)/(4*pad))+1} iterations.")

    # ---------------- cout ----------------
    say("\nPOURQUOI C'EST LENT")
    act = B * ic * S * S * 4 / 1e6
    say(f"  etat dual garde par couche : {act:.1f} Mo (batch {B}, float32)")
    say(f"  sur K = {K} couches : {K*act:.1f} Mo d'activations pour le backward")
    say(f"  chaque iteration = 2 convolutions PLEINE RESOLUTION ({S}x{S}) ;")
    say(f"  un UNet descend a {S//8}x{S//8} des le 3e etage et y fait l'essentiel du calcul.")

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
