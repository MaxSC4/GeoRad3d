# -------------------------------------------------------
# Génération d'un GIF des coupes 2D en Z (X-Y) avec un slider animé
# -------------------------------------------------------
from __future__ import annotations
from pathlib import Path
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

    finite_mask = np.isfinite(V)
    finite_vals = V[finite_mask]
    if finite_vals.size == 0:
        raise ValueError("Impossible de créer un GIF : volume sans valeurs finies.")
    if vmin is None:
        vmin = float(np.percentile(finite_vals, 2))
    if vmax is None:
        vmax = float(np.percentile(finite_vals, 98))

    def _prep_slice(slice_xy: np.ndarray) -> np.ndarray:
        arr = slice_xy
        if upsample_factor and upsample_factor != 1.0:
            arr = zoom(arr, upsample_factor, order=1)
        if gaussian_sigma and gaussian_sigma > 0:
            arr = gaussian_filter(arr, sigma=gaussian_sigma)
        return arr

    extent = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
    fig = plt.figure(figsize=(6.5, 7), dpi=dpi)
    gs = fig.add_gridspec(2, 1, height_ratios=[14, 1])
    ax = fig.add_subplot(gs[0])
    ax_slider = fig.add_subplot(gs[1])

    fig.suptitle(title, fontsize=12)

    k0 = 0
    img2d = _prep_slice(V[:, :, k0].T)

    im = ax.imshow(
        img2d,
        origin="lower",
        extent=extent,
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
    cursor_line = None
    if show_slider:
        ax_slider.set_xlim(0, len(zs) - 1)
        ax_slider.set_ylim(0, 1)
        ax_slider.axis("off")
        ax_slider.plot(  # ligne de fond
            [0, len(zs) - 1], [0.5, 0.5],
            color="lightgray", lw=4, alpha=0.7, solid_capstyle="round"
        )
        cursor_line, = ax_slider.plot([k0], [0.5], "o", color="tab:blue", markersize=8)
    else:
        ax_slider.axis("off")

    def _capture_frame() -> Image.Image:
        """Transforme la figure en image RGB sans passer par un disque/BytesIO."""
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return Image.fromarray(buf, mode="RGBA").convert("RGB")

    frames: list[Image.Image] = []
    frames.append(_capture_frame())

    # Boucle animation
    step = max(1, int(skip))
    for k in range(0, V.shape[2], step):
        if k == k0:
            continue
        sl = _prep_slice(V[:, :, k].T)

        im.set_data(sl)
        ax.set_title(f"Z = {zs[k]:.2f}  (index {k})")

        if show_slider:
            cursor_line.set_xdata([k])

        frames.append(_capture_frame())

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
