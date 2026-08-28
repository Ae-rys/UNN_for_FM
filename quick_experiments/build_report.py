# -*- coding: utf-8 -*-
"""
build_report.py
Assemble le document de synthese : lit rapport_sccp_unet.src.html et remplace
chaque <img src="chemin/local.png"> par la meme image encodee en data URI.

Pourquoi cette etape : une page publiee comme Artifact est servie sous une CSP
stricte qui bloque toute requete vers un autre hote ET tout fichier local. Une
image referencee par chemin relatif ne s'afficherait donc jamais. L'inlining est
la seule facon d'obtenir une page autonome — et il garde le fichier source
editable comme du HTML normal, au lieu d'un pate de base64.

La taille finale doit rester sous 16 Mo (base64 gonfle de ~33 %) ; le script
l'affiche et previent si on s'en approche.

Usage
-----
    source ~/.venvs/unn/bin/activate
    python make_report_figures.py     # (re)genere report_figs/
    python build_report.py            # -> rapport_sccp_unet.html
"""

import base64
import mimetypes
import os
import re

SRC = "rapport_sccp_unet.src.html"
DST = "rapport_sccp_unet.html"
LIMIT = 16 * 1024 * 1024


def inline(match):
    path = match.group(1)
    if path.startswith(("data:", "http://", "https://")):
        return match.group(0)
    if not os.path.exists(path):
        raise FileNotFoundError(f"image absente : {path} (lancer make_report_figures.py)")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    print(f"  inline {path} ({os.path.getsize(path)/1024:.0f} Ko -> {len(b64)/1024:.0f} Ko)",
          flush=True)
    return f'src="data:{mime};base64,{b64}"'


def main():
    with open(SRC, encoding="utf-8") as f:
        html = f.read()
    html = re.sub(r'src="([^"]+)"', inline, html)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(DST)
    print(f"\n-> {DST}  ({size/1024/1024:.2f} Mo)", flush=True)
    if size > LIMIT:
        print(f"   [ERREUR] au-dessus de la limite de {LIMIT/1024/1024:.0f} Mo", flush=True)
    elif size > 0.8 * LIMIT:
        print("   [attention] on approche de la limite de 16 Mo", flush=True)


if __name__ == "__main__":
    main()
