# -*- coding: utf-8 -*-
"""
flops_utils.py
Deux infos que tous les runs ecrivent desormais dans leur fichier de parametres
(`<run_dir>/parametres.txt`) :

  velocity_flops   FLOPs d'UNE evaluation de vitesse v_t(x) pour UN echantillon
                   (batch 1, forward seul, sans backward).
  train_time_s     temps total d'entrainement en secondes. Pour les runs pilotes
                   en steps avec reprise (CIFAR / AFHQ), c'est un CUMUL : il est
                   stocke dans latest.pt et repris a chaque redemarrage, donc il
                   totalise toutes les sessions du run, pas seulement la derniere.

Le comptage s'appuie sur torch.utils.flop_counter.FlopCounterMode, qui ne
comptabilise que les conv et les matmul (convention 1 MAC = 2 FLOPs). Les
operations elementwise (prox / soft-threshold, embeddings de t, normalisations)
ne sont PAS comptees : elles sont negligeables en FLOPs, meme quand elles
dominent le temps de calcul reel — cf. track_sccp_speed, l'ecart FLOPs/temps du
ScCP vient de l'efficacite GPU, pas du nombre d'operations. Ce chiffre sert donc
a comparer des architectures a cout arithmetique egal, pas a predire le temps.

Le fichier `params.txt` (un simple entier, lu tel quel par results/plot_losses.py)
reste inchange : tout passe par parametres.txt, format `cle=valeur`, dont les
lecteurs existants ne prennent que les cles qu'ils connaissent.
"""

import os
import time

import torch

PARAM_FILE = "parametres.txt"


# ---------------------------------------------------------------------------
# Comptage des FLOPs
# ---------------------------------------------------------------------------

def count_velocity_flops(model, call, quiet=False):
    """FLOPs (forward seul) d'un appel `call(model)`, qui doit representer UNE
    evaluation de vitesse sur UN echantillon.

    Le modele est bascule en eval() pendant la mesure (indispensable : les UNN
    avec use_checkpoint=True ne passent par torch.utils.checkpoint qu'en mode
    train, ce qui perturberait le comptage), puis remis dans son etat d'origine.

    Retourne un int, ou None si le comptage echoue (le comptage ne doit jamais
    faire tomber un entrainement).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    was_training = model.training
    model.eval()
    try:
        counter = FlopCounterMode(display=False)
        with torch.no_grad(), counter:
            call(model)
        return int(counter.get_total_flops())
    except Exception as exc:
        if not quiet:
            print(f"    [warn] comptage FLOPs impossible : {exc}", flush=True)
        return None
    finally:
        if was_training:
            model.train()


def flops_vector_model(model, dim, device, **call_kwargs):
    """Cas standard de nos UNN : model(cat([x_t, t], dim=-1)), entree (1, dim+1)."""
    xt_t = torch.zeros(1, dim + 1, device=device)
    return count_velocity_flops(model, lambda m: m(xt_t, **call_kwargs))


def flops_unet_model(model, img_shape, device):
    """Cas UNetModel (torchcfm) : model(t, x), avec x de forme (1, *img_shape)."""
    x = torch.zeros(1, *img_shape, device=device)
    t = torch.zeros(1, device=device)
    return count_velocity_flops(model, lambda m: m(t, x))


def format_flops(flops):
    if flops is None:
        return "n/a"
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if flops >= scale:
            return f"{flops / scale:.3f} {unit}FLOPs"
    return f"{flops} FLOPs"


def format_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Lecture / mise a jour de parametres.txt
# ---------------------------------------------------------------------------

def read_param_file(run_dir):
    """parametres.txt -> dict (ordre du fichier conserve). {} s'il n'existe pas."""
    path = os.path.join(run_dir, PARAM_FILE)
    fields = {}
    if not os.path.exists(path):
        return fields
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def update_param_file(run_dir, **fields):
    """Ecrit/met a jour des cles de parametres.txt sans toucher aux autres.

    Les cles deja presentes gardent leur position (pratique quand le fichier est
    reecrit a chaque checkpoint), les nouvelles sont ajoutees a la fin. Les
    valeurs None sont ignorees. Ecriture atomique (tmp + os.replace) : un kill
    pendant la reecriture ne laisse pas un fichier tronque.
    """
    os.makedirs(run_dir, exist_ok=True)
    current = read_param_file(run_dir)
    for key, value in fields.items():
        if value is not None:
            current[key] = value

    path = os.path.join(run_dir, PARAM_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for key, value in current.items():
            f.write(f"{key}={value}\n")
    os.replace(tmp, path)


def write_velocity_flops(run_dir, flops, note=None, verbose=True):
    """Enregistre les FLOPs d'une evaluation de vitesse (batch 1)."""
    update_param_file(
        run_dir,
        velocity_flops=flops if flops is not None else "n/a",
        velocity_flops_human=format_flops(flops),
        velocity_flops_note=note or "1 echantillon, forward seul, conv+matmul (torch FlopCounterMode)",
    )
    if verbose:
        print(f"FLOPs / eval. vitesse (batch 1) : {format_flops(flops)}", flush=True)
    return flops


def write_train_time(run_dir, seconds, steps=None, epochs=None):
    """Enregistre le temps total d'entrainement (cumule pour les runs repris)."""
    update_param_file(
        run_dir,
        train_time_s=f"{seconds:.1f}",
        train_time_human=format_duration(seconds),
        train_steps_done=steps,
        train_epochs_done=epochs,
        train_time_updated=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
