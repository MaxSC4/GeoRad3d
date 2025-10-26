# -------------------------------------------------------
# Point d'entrée du projet GeoRad3D
# - fit  : interpolation (krigeage 3D) + cross-validation
# - view : visualisation du volume + points
# - paint: peinture d'un .obj à partir du volume (vertex colors + texture/MTL optionnels)
# -------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from utils.animation import save_zslice_gif

# Logger (fallback simple si utils.logger absent)
try:
    from utils.logger import get_logger
except Exception:
    def get_logger(name: str):
        class _L:
            def info(self, *a, **k): print("[INFO]", *a)
            def warning(self, *a, **k): print("[WARN]", *a)
            def error(self, *a, **k): print("[ERROR]", *a)
        return _L()

from utils.load import load_points_csv
from utils.bounding import bbox_from_points
from utils.grid import regular_grid
from models.interpolation import krige_volume
from validation.crossval import loo_metrics
from pipelines.mesh_paint import paint_obj_with_volume

# Visualisation (optionnelle; la commande view s'en sert)
try:
    from utils.visualization import show_volume_and_points
    _HAS_VISU = True
except Exception:
    _HAS_VISU = False


# ---------- Répertoires par défaut ------------------------------------------
ROOT = Path(__file__).parent.resolve()
DATA = ROOT / "data"
OUT = ROOT / "outputs"
VOL_DIR = OUT / "volumes"
MESH_OUT = OUT / "meshes"
FIG_DIR = OUT / "figures"

for d in [OUT, VOL_DIR, MESH_OUT, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log = get_logger("main")


# ---------- Commandes --------------------------------------------------------

def cmd_fit(args: argparse.Namespace) -> None:
    log.info("Chargement des points…")
    df = load_points_csv(args.csv)
    x, y, z, r = df["X"].values, df["Y"].values, df["Z"].values, df["R"].values
    log.info(f"{len(df)} points chargés depuis {args.csv}")

    # BBox + grille
    bx, by, bz = bbox_from_points(x, y, z, pad_frac=args.pad)
    xs, ys, zs = regular_grid(bx, by, bz, nx=args.nx, ny=args.ny, nz=args.nz)
    log.info(f"Grille : nx={len(xs)}, ny={len(ys)}, nz={len(zs)}")

    # Krigeage 3D
    log.info(f"Krigeage 3D (variogram='{args.variogram}')…")
    est, var = krige_volume(
        xs, ys, zs,
        np.c_[x, y, z], r,
        variogram_model=args.variogram,
        variogram_parameters=None,
        enable_plotting=False
    )

    # Sauvegarde du volume
    vol_path = VOL_DIR / "volume.npz"
    np.savez_compressed(vol_path, xs=xs, ys=ys, zs=zs, est=est, var=var)
    log.info(f"Volume sauvegardé : {vol_path}")

    # Cross-validation LOO (optionnelle)
    if not args.skip_cv:
        log.info("Cross-validation Leave-One-Out…")
        cv = loo_metrics(np.c_[x, y, z], r, variogram_model=args.variogram, variogram_parameters=None)
        metrics = {k: cv[k] for k in ["ME", "RMSE", "MSSE", "VSE"]}
        pd.Series(metrics).to_csv(OUT / "validation_metrics.csv")
        (OUT / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log.info(f"Métriques LOO : {metrics}")
    else:
        log.info("Cross-validation SKIPPED (à la demande).")

    # Option : visu directe (rapide)
    if args.preview and _HAS_VISU:
        log.info("Prévisualisation du volume…")
        try:
            show_volume_and_points(xs, ys, zs, est, points_df=df, n_isosurfaces=3, cmap="viridis")
        except Exception as e:
            log.warning(f"Prévisualisation échouée : {e}")


def cmd_view(args: argparse.Namespace) -> None:
    if not _HAS_VISU:
        log.error("La visualisation n'est pas disponible (utils.visualization introuvable).")
        return
    vol_path = VOL_DIR / "volume.npz"
    if not vol_path.exists():
        log.error(f"Volume introuvable : {vol_path} — lance d'abord `fit`.")
        return

    data = np.load(vol_path)
    xs, ys, zs, est = data["xs"], data["ys"], data["zs"], data["est"]

    df = None
    if args.csv and Path(args.csv).exists():
        df = load_points_csv(args.csv)

    show_volume_and_points(
        xs, ys, zs, est,
        points_df=df,
        n_isosurfaces=args.nisos,
        vmin=args.vmin, vmax=args.vmax,
        cmap=args.cmap,
        title="Volume & Points",
        obj_path=args.obj,
        mesh_mode=args.mesh_mode,
        mesh_color="#9aa0a6",
        mesh_alpha=0.9,
        mesh_lw=0.6,
        mesh_max_edges=args.mesh_max_edges,
        show_isosurfaces=not args.no_isos,
    )


def cmd_paint(args: argparse.Namespace) -> None:
    vol_path = VOL_DIR / "volume.npz"
    if not vol_path.exists():
        log.error(f"Volume introuvable : {vol_path} — lance d'abord `fit`.")
        return

    data = np.load(vol_path)
    xs, ys, zs, est = data["xs"], data["ys"], data["zs"], data["est"]

    res = paint_obj_with_volume(
        obj_in=args.obj,
        xs=xs, ys=ys, zs=zs, volume=est,
        out_dir=args.out_dir or MESH_OUT,
        cmap_name=args.cmap,
        vmin=args.vmin, vmax=args.vmax,
        bake_texture_png=not args.no_texture,
        texture_size=(args.tex_w, args.tex_h),
        nan_color=None if args.no_nan_color else (128, 128, 128),
        write_stats_json=True
    )
    log.info(f"Peinture terminée : {res}")

def cmd_gif(args):
    vol_path = VOL_DIR / "volume.npz"
    if not vol_path.exists():
        log.error(f"Volume introuvable : {vol_path} — lance d'abord `fit`.")
        return
    data = np.load(vol_path)
    xs, ys, zs, est = data["xs"], data["ys"], data["zs"], data["est"]

    df = None
    if args.csv and Path(args.csv).exists():
        df = load_points_csv(args.csv)

    out = Path(args.out or (FIG_DIR / "z_sweep.gif"))
    out.parent.mkdir(parents=True, exist_ok=True)

    path = save_zslice_gif(
        xs, ys, zs, est,
        out_path=out,
        points_df=df,
        vmin=args.vmin, vmax=args.vmax,
        cmap=args.cmap,
        upsample_factor=args.upsample,
        gaussian_sigma=args.sigma,
        skip=args.skip,
        fps=args.fps,
        dpi=args.dpi,
        title="GeoRad3D — Z sweep"
    )
    log.info(f"GIF écrit : {path}")

# ---------- Parser CLI -------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("GeoRad3D — pipeline radioactivité 3D")
    sub = p.add_subparsers(dest="cmd", required=True)

    # fit
    pf = sub.add_parser("fit", help="Interpoler le volume (krigeage 3D) + cross-validation")
    pf.add_argument("--csv", default=str(DATA / "raw" / "points.csv"), help="CSV X,Y,Z,R")
    pf.add_argument("--pad", type=float, default=0.05, help="Marge relative de la bbox")
    pf.add_argument("--nx", type=int, default=60)
    pf.add_argument("--ny", type=int, default=60)
    pf.add_argument("--nz", type=int, default=40)
    pf.add_argument("--variogram", default="spherical", choices=["spherical", "exponential", "gaussian", "linear"])
    pf.add_argument("--preview", action="store_true", help="Ouvrir une visu rapide après le fit (si dispo)")
    pf.add_argument("--skip-cv", action="store_true", help="saute la cross-validation LOO")
    pf.set_defaults(func=cmd_fit)

    # view
    pv = sub.add_parser("view", help="Visualiser le volume + points")
    pv.add_argument("--csv", default=str(DATA / "raw" / "points.csv"), help="CSV des points (optionnel)")
    pv.add_argument("--nisos", type=int, default=3, help="Nombre d’isosurfaces")
    pv.add_argument("--vmin", type=float, default=None, help="Borne basse des couleurs")
    pv.add_argument("--vmax", type=float, default=None, help="Borne haute des couleurs")
    pv.add_argument("--cmap", default="viridis")
    pv.add_argument("--obj", default=None, help="Maillage .obj à superposer")
    pv.add_argument("--no-isos", action="store_true", help="Désactiver les isosurfaces (recommandé avec .obj)")
    pv.add_argument("--mesh-mode", default="wire", choices=["wire","surface"], help="Affichage du mesh")
    pv.add_argument("--mesh-max-edges", type=int, default=80000, help="Max edges wireframe")
    pv.set_defaults(func=cmd_view)

    # paint
    pp = sub.add_parser("paint", help="Peindre un .obj depuis le volume")
    pp.add_argument("--obj", default=str(DATA / "meshes" / "affleurement.obj"), help="OBJ à peindre")
    pp.add_argument("--out-dir", default=str(MESH_OUT), help="Répertoire de sortie")
    pp.add_argument("--cmap", default="viridis")
    pp.add_argument("--vmin", type=float, default=None)
    pp.add_argument("--vmax", type=float, default=None)
    pp.add_argument("--no-texture", action="store_true", help="Désactiver le baking PNG/MTL (vertex colors uniquement)")
    pp.add_argument("--tex-w", type=int, default=2048, help="Largeur texture")
    pp.add_argument("--tex-h", type=int, default=2048, help="Hauteur texture")
    pp.add_argument("--no-nan-color", action="store_true", help="Ne pas recolorer les NaN (sommets hors volume)")
    pp.set_defaults(func=cmd_paint)

    pg = sub.add_parser("gif", help="Générer un GIF des coupes 2D sur Z")
    pg.add_argument("--csv", default=str(DATA / "raw" / "points.csv"), help="Points (pour superposer les mesures)")
    pg.add_argument("--out", default=str(FIG_DIR / "z_sweep.gif"))
    pg.add_argument("--cmap", default="viridis")
    pg.add_argument("--vmin", type=float, default=None)
    pg.add_argument("--vmax", type=float, default=None)
    pg.add_argument("--upsample", type=float, default=2.0, help="Upsample factor (>=1.0)")
    pg.add_argument("--sigma", type=float, default=0.8, help="Gaussian blur")
    pg.add_argument("--skip", type=int, default=1, help="Stride sur Z (1=toutes les tranches)")
    pg.add_argument("--fps", type=int, default=10)
    pg.add_argument("--dpi", type=int, default=120)
    pg.set_defaults(func=cmd_gif)

    return p


# ---------- Entrée -----------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
