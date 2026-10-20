"""Build a browsable gallery of the objective-champion GIFs.

Outputs two artifacts next to the GIFs:
  * ``figures/gifs/gallery.html`` -- a self-contained page that plays every GIF
    with its objective name, validated fitness, and description (relative refs;
    open it directly in a browser from the repo).
  * ``figures/gif_contact_sheet.png`` -- a static montage of three sampled frames
    (early / middle / late) per objective for papers and quick preview.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
GIFS = ROOT / "figures" / "gifs"


def _sampled_frames(gif_path, k=3):
    im = Image.open(gif_path)
    frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(im)]
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [frames[i] for i in idx]


def contact_sheet(manifest: pd.DataFrame, out: Path, k=3):
    rows = len(manifest)
    fig, axes = plt.subplots(rows, k, figsize=(2.6 * k, 2.6 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r, (_, m) in enumerate(manifest.iterrows()):
        frames = _sampled_frames(ROOT / m["gif"], k)
        for c in range(k):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(frames):
                ax.imshow(frames[c])
            if c == 0:
                ax.set_title(f"{m['objective']}  (fit={m['fitness']:.3f})",
                             loc="left", fontsize=9, color="black")
    fig.suptitle("SPP objective-champion behaviors: early / middle / late frame",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out, dpi=110)
    plt.close(fig)


HTML_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPP behavioral repertoire — objective-champion GIFs</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0c1020; color:#e8edf6;
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { padding:28px 32px 8px; }
  h1 { margin:0 0 6px; font-size:22px; }
  p.sub { margin:0; color:#9aa7c2; max-width:70ch; line-height:1.45; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
          gap:20px; padding:24px 32px 48px; }
  .card { background:#141a30; border:1px solid #263053; border-radius:12px;
          overflow:hidden; display:flex; flex-direction:column; }
  .card img { width:100%; display:block; background:#0c1020; }
  .meta { padding:12px 14px 16px; }
  .name { font-weight:600; font-size:15px; text-transform:capitalize; }
  .fit  { float:right; font-variant-numeric:tabular-nums; color:#5ec8ff; font-weight:600; }
  .tag  { display:inline-block; margin-top:6px; font-size:11px; color:#0c1020;
          background:#7fd08a; border-radius:6px; padding:1px 7px; }
  .tag.two { background:#f0a860; }
  .blurb { margin-top:8px; color:#aab6d4; font-size:13px; line-height:1.4; }
</style></head><body>
<header>
<h1>Self-propelled particles — highest-fitness behaviors per objective</h1>
<p class="sub">Each animation is the parameter set with the highest validated
fitness found for its objective function, pooled over the classical QD archives,
the density-expanded corpus, and a fresh directed refinement search (density up
to &rho;=0.70). Arrow hue encodes heading direction; the yellow trail is the
group center of mass. Two-species runs color the two particle types.</p>
</header>
<div class="grid">
"""


def gallery_html(manifest: pd.DataFrame, out: Path):
    cards = []
    for _, m in manifest.sort_values("kind").iterrows():
        name = str(m["objective"]).replace("_", " ")
        tag = "two-species" if m["kind"] == "two" else "single-species"
        cls = "tag two" if m["kind"] == "two" else "tag"
        gif_rel = Path(m["gif"]).name  # gallery lives beside the gifs
        cards.append(
            f'<div class="card"><img src="{gif_rel}" alt="{name}" loading="lazy">'
            f'<div class="meta"><div><span class="fit">{m["fitness"]:.3f}</span>'
            f'<span class="name">{name}</span></div>'
            f'<span class="{cls}">{tag}</span>'
            f'<div class="blurb">{m["blurb"]}</div></div></div>')
    html = HTML_HEAD + "\n".join(cards) + "\n</div></body></html>"
    out.write_text(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="",
                    help="output-set suffix, e.g. 'periodic' (matches discover_and_animate --tag)")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    manifest_path = ROOT / "results" / "expanded" / f"gif_manifest{suffix}.csv"
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path}; run discover_and_animate.py first")
    manifest = pd.read_csv(manifest_path).sort_values("fitness", ascending=False).reset_index(drop=True)
    gallery = GIFS / f"gallery{suffix}.html"
    sheet = ROOT / "figures" / f"fig_gif_contact_sheet{suffix}.png"
    gallery_html(manifest, gallery)
    contact_sheet(manifest, sheet)
    print("wrote", gallery)
    print("wrote", sheet)


if __name__ == "__main__":
    main()
