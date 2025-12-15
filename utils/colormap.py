import numpy as np
from matplotlib import cm


def values_to_rgb(
    vals: np.ndarray,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    clip: bool = True
):
    """
    Mappe des valeurs scalaires vers des couleurs RGB 8 bits (0–255),
    selon une colormap Matplotlib perceptuelle (viridis, plasma, etc.).

    Paramètres
    ----------
    vals : np.ndarray
        Tableau 1D ou N×1 des valeurs à convertir.
    cmap_name : str
        Nom de la colormap Matplotlib (ex: 'viridis', 'plasma', 'magma').
    vmin, vmax : float, optionnel
        Bornes d’échelle. Si None → percentiles 2 et 98.
    clip : bool
        Si True, les valeurs en dehors de [vmin, vmax] sont bornées.

    Retour
    ------
    rgb : np.ndarray
        Tableau (N,3) des couleurs RGB en uint8 (0–255).
    (vmin_used, vmax_used) : tuple[float, float]
        Bornes réellement utilisées pour le mapping.
    """
    vals = np.asarray(vals, dtype=float)
    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size == 0:
        raise ValueError("Aucune valeur finie fournie pour le mapping de couleurs.")

    # Détermination automatique des bornes
    if vmin is None:
        vmin = float(np.percentile(finite_vals, 2))
    if vmax is None:
        vmax = float(np.percentile(finite_vals, 98))
    if np.isclose(vmax, vmin):
        vmax = vmin + 1e-6  # évite division par zéro

    # Échelle normalisée entre 0 et 1
    if clip:
        vals = np.clip(vals, vmin, vmax)

    normed = (vals - vmin) / (vmax - vmin + 1e-12)
    normed = np.nan_to_num(normed, nan=0.0, posinf=1.0, neginf=0.0, copy=False)
    cmap = cm.get_cmap(cmap_name)

    # Conversion RGBA → RGB [0..255]
    rgba = cmap(normed)  # shape (N,4)
    rgb255 = (rgba[:, :3] * 255).astype(np.uint8)

    return rgb255, (vmin, vmax)
