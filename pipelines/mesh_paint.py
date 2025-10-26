# -------------------------------------------------------
# Pipeline : peindre un .obj à partir d'un volume 3D interpolé
# Étapes :
#   1) Charger l'OBJ (sommets/UV/faces)
#   2) Échantillonner le volume aux sommets (trilin)
#   3) Mapper valeurs -> couleurs RGB (colormap perceptuelle)
#   4) Écrire un OBJ "vertex-colored"
#   5) (option) Baker une texture PNG via UV + écrire un MTL et le brancher
# -------------------------------------------------------
from __future__ import annotations
from pathlib import Path
import json
import numpy as np

# Logger : utilise utils.logger.get_logger si dispo, sinon fallback print
try:
    from utils.logger import get_logger
except Exception:
    def get_logger(name: str):
        class _L:
            def info(self, *a, **k): print("[INFO]", *a)
            def warning(self, *a, **k): print("[WARN]", *a)
            def error(self, *a, **k): print("[ERROR]", *a)
        return _L()

from utils.obj_io import SimpleOBJ
from utils.colormap import values_to_rgb
from models.sampler3d import ScalarFieldSampler
from exporters.texture_bake import bake_texture
from exporters.mtl_writer import write_new_mtl, update_obj_usemtl


def paint_obj_with_volume(
    obj_in: str | Path,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    volume: np.ndarray,                  # volume 3D interpolé (X,Y,Z) ou (Z,Y,X)
    out_dir: str | Path,
    cmap_name: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    bake_texture_png: bool = True,
    texture_size: tuple[int, int] = (2048, 2048),
    nan_color: tuple[int, int, int] | None = (128, 128, 128),  # couleur des sommets hors bbox (NaN)
    write_stats_json: bool = True
) -> dict:
    """
    Peint un .obj avec les valeurs du volume 3D.

    Paramètres
    ----------
    obj_in : str | Path
        Chemin du fichier OBJ d'entrée.
    xs, ys, zs : np.ndarray
        Axes 1D de la grille régulière (croissants).
    volume : np.ndarray
        Volume 3D interpolé (peut être (nx,ny,nz) ou (nz,ny,nx); auto géré).
    out_dir : str | Path
        Dossier de sortie (sera créé).
    cmap_name : str
        Colormap Matplotlib (ex: 'viridis', 'plasma').
    vmin, vmax : float | None
        Bornes du mapping couleurs (None → percentiles 2/98 des valeurs échantillonnées).
    bake_texture_png : bool
        Si True, génère une texture PNG via UV + un MTL et met à jour l'OBJ.
    texture_size : (int,int)
        Taille de la texture PNG (pixels).
    nan_color : (int,int,int) | None
        Couleur à utiliser pour les NaN (sommets hors volume). None pour laisser tel quel.
    write_stats_json : bool
        Si True, écrit un fichier JSON de stats à côté des sorties.

    Retour
    ------
    dict : chemins de sortie + quelques stats
        {
          "obj_out": ".../xxx_painted.obj",
          "mtl_out": ".../xxx_painted.mtl" | None,
          "texture_out": ".../xxx_texture.png" | None,
          "stats": {...}
        }
    """
    logger = get_logger("mesh_paint")
    obj_in = Path(obj_in)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Charger le mesh
    obj = SimpleOBJ.load(obj_in)
    n_verts = int(obj.vertices.shape[0])
    n_faces = int(len(obj.faces_v))
    logger.info(f"OBJ chargé : {n_verts} sommets, {n_faces} faces, {obj.uvs.shape[0]} UV")

    if n_verts == 0:
        raise ValueError(f"Aucun sommet dans l'OBJ : {obj_in}")

    # 2) Échantillonnage du volume aux sommets
    sampler = ScalarFieldSampler(xs, ys, zs, volume)
    vals = sampler.sample_xyz(obj.vertices)  # (N,)
    n_nan = int(np.isnan(vals).sum())
    logger.info(f"Échantillonnage terminé. Sommets hors grille (NaN) : {n_nan}/{n_verts}")

    # 3) Mapping valeurs -> RGB
    rgb, (used_vmin, used_vmax) = values_to_rgb(vals, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
    if n_nan > 0 and nan_color is not None:
        mask_nan = np.isnan(vals)
        rgb[mask_nan] = np.array(nan_color, dtype=np.uint8)

    # 4) Sauvegarde OBJ vertex-colored
    obj_out_path = out_dir / f"{obj_in.stem}_painted.obj"
    obj.save_with_vertex_colors(obj_out_path, rgb)
    logger.info(f"OBJ écrit (vertex colors) : {obj_out_path}")

    # 5) Option : baking UV -> PNG + MTL
    tex_out_path = None
    mtl_out_path = None
    if bake_texture_png:
        tex_w, tex_h = texture_size
        img = bake_texture(obj.uvs, obj.faces_vt, rgb, obj.faces_v, tex_w=tex_w, tex_h=tex_h)
        if img is not None:
            tex_out_path = out_dir / f"{obj_in.stem}_texture.png"
            img.save(tex_out_path)
            mtl_out_path = out_dir / f"{obj_in.stem}_painted.mtl"
            write_new_mtl(mtl_out_path, texture_name=tex_out_path.name, material_name="Radioactivity")
            update_obj_usemtl(obj_out_path, mtllib_name=mtl_out_path.name, usemtl_name="Radioactivity")
            logger.info(f"Texture + MTL écrits : {tex_out_path}, {mtl_out_path}")
        else:
            logger.warning("Pas d'UV utilisables → texture/MTL non générés (utiliser l'OBJ vertex-colored).")

    # Stats + traçabilité
    stats = {
        "input_obj": str(obj_in),
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "n_nan": n_nan,
        "cmap": cmap_name,
        "vmin_used": float(used_vmin),
        "vmax_used": float(used_vmax),
        "baked_texture": bool(tex_out_path is not None),
        "texture_size": list(texture_size) if tex_out_path is not None else None,
    }

    if write_stats_json:
        stats_path = out_dir / f"{obj_in.stem}_paint_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats écrites : {stats_path}")

    return {
        "obj_out": str(obj_out_path),
        "mtl_out": str(mtl_out_path) if mtl_out_path else None,
        "texture_out": str(tex_out_path) if tex_out_path else None,
        "stats": stats,
    }
