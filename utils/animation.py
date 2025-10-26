# -------------------------------------------------------
# Génération d'un GIF des coupes 2D en Z (X-Y) avec un slider animé
# -------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, zoom
from PIL import Image


def _ensure_xyz_order(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, vol: np.ndarray):
    """Assure shape (len(xs), len(ys), len(zs)); transpose si (Z,Y,X)."""
    sx, sy, sz = len(xs), len(ys), len(zs)
    if vol.shape == (sx, sy, sz):
        return vol
    if vol.shape == (sz, sy, sx):
        return np.transpose(vol, (2, 1, 0))
    raise ValueError(
        f"Volume shape {vol.shape} incompatible avec axes "
        f"({sx},{sy},{sz}) ou ({sz},{sy},{sx})."
    )


def save_zslice_gif(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    volume: np.ndarray,
    out_path: str | Path,
    points_df=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    upsample_factor: float = 2.0,
    gaussian_sigma: float = 0.8,
    skip: int = 1,
    fps: int = 10,
    dpi: int = 120,
    title: str = "Radioactivity — Z sweep",
    show_slider: bool = True,
) -> str:
    """
    Crée un GIF montrant l'évolution de la radioactivité sur toutes les coupes Z,
    avec un slider animé simulé en bas de la figure.
    """
    out_path = str(Path(out_path))
    V = _ensure_xyz_order(xs, ys, zs, volume)

    finite_vals = V[np.isfinite(V)]
    if vmin is None:
        vmin = float(np.percentile(finite_vals, 2))
    if vmax is None:
        vmax = float(np.percentile(finite_vals, 98))

    fig = plt.figure(figsize=(6.5, 7), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[14, 1])
    ax = fig.add_subplot(gs[0])
    ax_slider = fig.add_subplot(gs[1])

    fig.suptitle(title, fontsize=12)

    k0 = 0
    img2d = V[:, :, k0].T
    if upsample_factor and upsample_factor != 1.0:
        img2d = zoom(img2d, upsample_factor, order=1)
    if gaussian_sigma and gaussian_sigma > 0:
        img2d = gaussian_filter(img2d, sigma=gaussian_sigma)

    im = ax.imshow(
        img2d,
        origin="lower",
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Radioactivity (cpm)")

    if points_df is not None and len(points_df) > 0:
        x = points_df["X"].values
        y = points_df["Y"].values
        ax.scatter(x, y, s=60, c="k", zorder=5)
        ax.scatter(x, y, s=30, c="#00FFFF", zorder=6)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Z = {zs[k0]:.2f}  (index {k0})")

    # --- Faux slider ---
    ax_slider.set_xlim(0, len(zs) - 1)
    ax_slider.set_ylim(0, 1)
    ax_slider.axis("off")
    # ligne de fond
    ax_slider.plot([0, len(zs) - 1], [0.5, 0.5], color="lightgray", lw=4, alpha=0.7, solid_capstyle="round")
    # curseur mobile
    cursor_line, = ax_slider.plot([k0], [0.5], "o", color="tab:blue", markersize=8)

    # Conversion figure → image
    def _fig_to_pil() -> Image.Image:
        buf = io.BytesIO()
        fig.canvas.draw()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        buf.close()
        return img

    frames: list[Image.Image] = []
    frames.append(_fig_to_pil())

    # Boucle animation
    for k in range(0, V.shape[2], max(1, int(skip))):
        if k == k0:
            continue
        sl = V[:, :, k].T
        if upsample_factor and upsample_factor != 1.0:
            sl = zoom(sl, upsample_factor, order=1)
        if gaussian_sigma and gaussian_sigma > 0:
            sl = gaussian_filter(sl, sigma=gaussian_sigma)

        im.set_data(sl)
        ax.set_title(f"Z = {zs[k]:.2f}  (index {k})")

        if show_slider:
            cursor_line.set_xdata([k])

        frames.append(_fig_to_pil())

    plt.close(fig)

    # Sauvegarde GIF
    duration_ms = int(1000 / max(1, fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
    )

    return out_path
