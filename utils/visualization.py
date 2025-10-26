# -------------------------------------------------------
# Visualisation 3D/2D du volume interpolé + points d'observation
# - Vue 3D : isosurfaces via marching_cubes (si dispo) OU fallback scatter
# - Vue 2D : coupe Z avec slider, upsample + gaussian blur (optionnel)
# - Tooltips 2D avec mplcursors (si dispo), clic 3D → label
# - Patchs perf : downsample "fast", step_size marching_cubes, décimation trimesh
# - Compat NumPy 2.x : np.ptp(...) au lieu de ndarray.ptp()
# -------------------------------------------------------
from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Optionnels (recommandés)
try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import mplcursors  # noqa: F401
    _HAS_MPLCURSORS = True
except Exception:
    _HAS_MPLCURSORS = False

from scipy.ndimage import gaussian_filter, zoom


# ---------- Helpers ----------------------------------------------------------

def _ensure_xyz_order(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, vol: np.ndarray):
    """
    S'assure que 'vol' est de shape (len(xs), len(ys), len(zs)).
    Si le volume est (len(zs), len(ys), len(xs)), on le transpose.
    """
    sx, sy, sz = len(xs), len(ys), len(zs)
    if vol.shape == (sx, sy, sz):
        return vol
    if vol.shape == (sz, sy, sx):
        return np.transpose(vol, (2, 1, 0))
    raise ValueError(
        f"Shape du volume {vol.shape} incompatible avec axes "
        f"({sx}, {sy}, {sz}) (X,Y,Z) ou ({sz}, {sy}, {sx}) (Z,Y,X)."
    )


def _slice_Z(volume_xyz: np.ndarray, k: int):
    """Retourne la coupe 2D (X,Y) au niveau d'index Z=k."""
    # transpose pour imshow avec extent (x_min, x_max, y_min, y_max)
    return volume_xyz[:, :, k].T


def _compute_isolevels(volume_xyz: np.ndarray, n_isos: int, vmin: float | None, vmax: float | None):
    """
    Calcule des isovaleurs soit par percentiles (si vmin/vmax None),
    soit réparties linéairement entre vmin/vmax.
    """
    vv = volume_xyz[np.isfinite(volume_xyz)]
    if vmin is None:
        vmin = float(np.percentile(vv, 10))
    if vmax is None:
        vmax = float(np.percentile(vv, 90))
    levels = np.linspace(vmin, vmax, n_isos + 2)[1:-1]  # on évite les extrêmes exacts
    return levels, vmin, vmax


def _plot_isosurface(
    ax3d,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    volume_xyz: np.ndarray,
    level: float,
    face_alpha: float = 0.25,
    cmap: str = "viridis",
    mc_step: int = 2,
    decimate_max_faces: int | None = 60000,
    fallback_scatter_step: int = 3,
):
    """
    Trace une isosurface via marching_cubes (si dispo). Sinon, fallback en scatter 3D
    des voxels proches du niveau (avec sous-échantillonnage).
    - mc_step : pas d'échantillonnage marching_cubes (2 ou 3 -> beaucoup plus rapide)
    - decimate_max_faces : si trimesh dispo, essaie de réduire à ~ce nombre de faces
    - fallback_scatter_step : stride pour sous-échantillonner les voxels en fallback
    """
    # Normalisation couleur pour cet iso
    denom = np.ptp(volume_xyz) + 1e-12
    iso_color = plt.get_cmap(cmap)((level - np.nanmin(volume_xyz)) / denom)

    if _HAS_SKIMAGE:
        # marching_cubes attend (Z,Y,X) et spacing (dz,dy,dx)
        vol_zyx = np.transpose(volume_xyz, (2, 1, 0))
        dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
        dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
        dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
        verts, faces, _, _ = marching_cubes(
            vol_zyx, level=level, spacing=(dz, dy, dx),
            step_size=int(max(1, mc_step))
        )
        # coords en (z,y,x) -> (x,y,z)
        verts_xyz = np.stack([verts[:, 2] + xs[0], verts[:, 1] + ys[0], verts[:, 0] + zs[0]], axis=1)

        # Décimation (si trimesh dispo)
        if decimate_max_faces is not None:
            try:
                import trimesh
                mesh_tri = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
                if len(mesh_tri.faces) > decimate_max_faces:
                    mesh_tri = mesh_tri.simplify_quadratic_decimation(decimate_max_faces)
                faces = mesh_tri.faces
                verts_xyz = mesh_tri.vertices
            except Exception:
                # pas de décimation possible -> on garde tel quel
                pass

        mesh = Poly3DCollection(verts_xyz[faces], alpha=face_alpha, linewidths=0, antialiased=False)
        mesh.set_edgecolor("none")
        mesh.set_facecolor(iso_color)
        ax3d.add_collection3d(mesh)

    else:
        # Fallback : scatter des voxels "proches" du niveau, sous-échantillonné
        tol = (np.nanmax(volume_xyz) - np.nanmin(volume_xyz)) * 0.02
        mask = np.isfinite(volume_xyz) & (np.abs(volume_xyz - level) < tol)
        if not np.any(mask):
            return
        # Sous-échantillonnage fort pour fluidité
        mask_ds = np.zeros_like(mask)
        mask_ds[::fallback_scatter_step, ::fallback_scatter_step, ::fallback_scatter_step] = \
            mask[::fallback_scatter_step, ::fallback_scatter_step, ::fallback_scatter_step]
        if not np.any(mask_ds):
            return
        xs3d, ys3d, zs3d = np.meshgrid(xs, ys, zs, indexing="xy")
        X = xs3d[mask_ds]; Y = ys3d[mask_ds]; Z = zs3d[mask_ds]
        ax3d.scatter(X, Y, Z, s=1, alpha=0.15, c=[iso_color])


def _scatter_points_3d(ax3d, df_points, neon_color="#00FFFF", s=14, halo=1.8):
    """
    Affiche les points d'observation en 3D avec halo noir + couleur néon.
    df_points: pandas DataFrame avec col. X,Y,Z,R (R optionnelle pour labels)
    """
    x = df_points["X"].values
    y = df_points["Y"].values
    z = df_points["Z"].values

    # Halo noir
    ax3d.scatter(x, y, z, s=s * halo, c="k", alpha=1.0, depthshade=False, zorder=5)
    # Couleur néon (cyan par défaut)
    pts = ax3d.scatter(x, y, z, s=s, c=neon_color, alpha=1.0, depthshade=False, zorder=6)

    # Click → label
    def _on_pick(event):
        ind = event.ind[0]
        if "R" in df_points.columns:
            label = f"R={df_points['R'].values[ind]:.3g} @ ({x[ind]:.2f}, {y[ind]:.2f}, {z[ind]:.2f})"
        else:
            label = f"({x[ind]:.2f}, {y[ind]:.2f}, {z[ind]:.2f})"
        ax3d.text(
            x[ind], y[ind], z[ind], label, color="w",
            bbox=dict(boxstyle="round,pad=0.2", fc="k", ec="none", alpha=0.65)
        )
        ax3d.figure.canvas.draw_idle()

    pts.set_picker(True)
    ax3d.figure.canvas.mpl_connect("pick_event", _on_pick)
    return pts


def _scatter_points_2d(ax2d, df_points, neon_color="#00FFFF", s=30, halo=2.2):
    """
    Affiche les points sur la coupe 2D (X,Y) avec halo + tooltips si mplcursors dispo.
    """
    x = df_points["X"].values
    y = df_points["Y"].values

    ax2d.scatter(x, y, s=s * halo, c="k", alpha=1.0, zorder=5)
    sc = ax2d.scatter(x, y, s=s, c=neon_color, alpha=1.0, zorder=6)

    if _HAS_MPLCURSORS:
        import mplcursors
        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _(sel):
            # --- récupération robuste de l'index ---
            i = getattr(sel, "index", None)  # préférence : sel.index (mplcursors >= 0.5)
            if i is None:
                # Fallback : nearest-neighbour sur les offsets (compatible MaskedArray)
                xy = sc.get_offsets()
                # sel.target est (x, y) dans les données
                dx = xy[:, 0] - sel.target[0]
                dy = xy[:, 1] - sel.target[1]
                i = int(np.argmin(dx * dx + dy * dy))
            # ---------------------------------------

            if "R" in df_points.columns:
                sel.annotation.set_text(
                    f"R={df_points['R'].values[i]:.3g}\nX={x[i]:.2f}\nY={y[i]:.2f}"
                )
            else:
                sel.annotation.set_text(f"X={x[i]:.2f}\nY={y[i]:.2f}")
            sel.annotation.get_bbox_patch().set(
                fc="k", ec="none", alpha=0.75, boxstyle="round,pad=0.2"
            )

    return sc


# ---------- API principale ---------------------------------------------------

def show_volume_and_points(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    volume: np.ndarray,
    points_df=None,
    n_isosurfaces: int = 3,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    initial_k: int | None = None,
    upsample_factor: float = 2.0,
    gaussian_sigma: float = 0.8,
    title: str = "Radioactivity — 3D isosurfaces & Z-slice",
    # nouveaux paramètres perf :
    fast: bool = True,
    downsample_factor: float = 0.5,
    mc_step: int = 2,
    decimate_max_faces: int | None = 60000,
    fallback_scatter_step: int = 3,
):
    """
    Affiche une fenêtre avec :
      - à gauche : vue 3D isosurfaces + points
      - à droite : coupe 2D en Z avec slider pour changer la tranche

    Paramètres clés pour la fluidité :
      fast=True, downsample_factor=0.5 → downsample du volume pour l'affichage
      mc_step=2 → moins de triangles aux isosurfaces
      decimate_max_faces=60000 → tente une décimation via trimesh
      fallback_scatter_step=3 → sous-échantillonne le scatter fallback
    """
    # Harmonise l'ordre (X,Y,Z)
    V = _ensure_xyz_order(xs, ys, zs, volume)

    # Downsample "affichage"
    if fast and downsample_factor != 1.0:
        V = zoom(V, downsample_factor, order=1)  # bilinear 3D
        xs = np.linspace(xs.min(), xs.max(), V.shape[0])
        ys = np.linspace(ys.min(), ys.max(), V.shape[1])
        zs = np.linspace(zs.min(), zs.max(), V.shape[2])

    nx, ny, nz = V.shape

    # Isovaleurs
    levels, used_vmin, used_vmax = _compute_isolevels(V, n_isosurfaces, vmin, vmax)

    # Figure : 2 colonnes + slider
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=14)

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    # ---- Vue 3D : isosurfaces
    for lv in levels:
        _plot_isosurface(
            ax3d, xs, ys, zs, V, level=lv, face_alpha=0.25, cmap=cmap,
            mc_step=mc_step, decimate_max_faces=decimate_max_faces,
            fallback_scatter_step=fallback_scatter_step
        )

    # Points 3D
    if points_df is not None and len(points_df) > 0:
        _scatter_points_3d(ax3d, points_df, neon_color="#00FFFF")

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"Isosurfaces ({len(levels)} niveaux)\n[{used_vmin:.3g} .. {used_vmax:.3g}]")
    ax3d.view_init(elev=25, azim=-65)
    ax3d.set_xlim(xs.min(), xs.max())
    ax3d.set_ylim(ys.min(), ys.max())
    ax3d.set_zlim(zs.min(), zs.max())

    # ---- Vue 2D : coupe Z
    if initial_k is None:
        initial_k = nz // 2
    img2d = _slice_Z(V, initial_k)

    # Upsample + blur
    if upsample_factor and upsample_factor != 1.0:
        img2d = zoom(img2d, upsample_factor, order=1)  # bilinear
    if gaussian_sigma and gaussian_sigma > 0:
        img2d = gaussian_filter(img2d, sigma=gaussian_sigma)

    im = ax2d.imshow(
        img2d,
        origin="lower",
        extent=(xs.min(), xs.max(), ys.min(), ys.max()),
        vmin=used_vmin if vmin is None else vmin,
        vmax=used_vmax if vmax is None else vmax,
        cmap=cmap,
        aspect="equal",
    )
    cbar = plt.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
    cbar.set_label("Radioactivity (cpm)")

    # Points 2D
    if points_df is not None and len(points_df) > 0:
        _scatter_points_2d(ax2d, points_df, neon_color="#00FFFF")

    ax2d.set_title(f"Coupe Z @ index {initial_k} (Z={zs[initial_k]:.2f})")
    ax2d.set_xlabel("X")
    ax2d.set_ylabel("Y")

    # ---- Slider pour changer k
    ax_slider = plt.axes([0.56, 0.08, 0.35, 0.03])
    slider = Slider(ax_slider, "Z index", 0, nz - 1, valinit=initial_k, valfmt="%0.0f")

    def _update(val):
        k = int(slider.val)
        sl = _slice_Z(V, k)
        if upsample_factor and upsample_factor != 1.0:
            sl = zoom(sl, upsample_factor, order=1)
        if gaussian_sigma and gaussian_sigma > 0:
            sl = gaussian_filter(sl, sigma=gaussian_sigma)
        im.set_data(sl)
        ax2d.set_title(f"Coupe Z @ index {k} (Z={zs[k]:.2f})")
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


# ---------- Diagnostics validation ------------------------------------------

def plot_crossval_diagnostics(
    obs: np.ndarray, pred: np.ndarray, var: np.ndarray | None = None,
    title: str = "Cross-validation diagnostics"
):
    """
    Graphiques : Pred vs Obs, résidus vs préd, et histogramme des résidus standardisés si 'var' est fourni.
    """
    from scipy.stats import norm

    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = obs - pred

    fig, axes = plt.subplots(1, 3 if var is not None else 2, figsize=(14, 4))
    axes = np.atleast_1d(axes)

    # 1) Pred vs Obs
    ax = axes[0]
    ax.scatter(obs, pred, s=20, alpha=0.7)
    mn, mx = np.nanmin([obs, pred]), np.nanmax([obs, pred])
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Observed")

    # 2) Résidus
    ax = axes[1]
    ax.scatter(pred, resid, s=20, alpha=0.7)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (obs - pred)")
    ax.set_title("Residuals")

    # 3) Résidus standardisés
    if var is not None and len(axes) > 2:
        ax = axes[2]
        z = resid / np.sqrt(np.maximum(var, 1e-12))
        ax.hist(z, bins=24, density=True, alpha=0.7)
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, norm.pdf(xs), "k--", lw=1, label="N(0,1)")
        ax.set_title("Standardized residuals")
        ax.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
