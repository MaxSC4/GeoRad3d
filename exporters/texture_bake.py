from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw


def _rasterize_triangle(img_draw: ImageDraw.ImageDraw, pts: np.ndarray, cols: np.ndarray) -> None:
    """
    Rasterisation simplifiée d’un triangle en espace UV (pixels).
    On remplit le polygone avec la moyenne des couleurs des sommets.

    Paramètres
    ----------
    img_draw : PIL.ImageDraw
        Context de dessin.
    pts : np.ndarray
        3x2 points du triangle en pixels (u_px, v_px) float32/float64.
    cols : np.ndarray
        3x3 couleurs RGB uint8 pour chaque sommet.
    """
    mean_col = tuple(np.mean(cols, axis=0).astype(np.uint8).tolist())
    img_draw.polygon([tuple(p) for p in pts], fill=mean_col)


def bake_texture(
    uvs: np.ndarray,
    faces_vt: list[list[int] | None],
    vertex_rgb: np.ndarray,
    faces_v: list[list[int]],
    tex_w: int = 2048,
    tex_h: int = 2048,
    bg: tuple[int, int, int] = (10, 10, 10)
):
    """
    Gènère une image de texture (PIL.Image) en projetant les couleurs
    par sommet sur les triangles UV. Si les UV n'existent pas ou sont
    inutilisables, retourne None.

    Paramètres
    ----------
    uvs : np.ndarray
        Tableau (M,2) des coordonnées UV ∈ [0,1].
    faces_vt : list
        Liste parallèle à faces_v : indices UV par face, ou None.
    vertex_rgb : np.ndarray
        Couleurs par sommet (N,3) uint8.
    faces_v : list
        Indices sommets par face (triangles de préférence).
    tex_w, tex_h : int
        Taille de la texture en pixels.
    bg : tuple[int,int,int]
        Couleur de fond de la texture.

    Retour
    ------
    PIL.Image | None
        L'image de texture générée, ou None si UV inutilisables.
    """
    # Si pas d'UV ou si au moins une face n'a pas de mapping UV → pas de baking
    if uvs is None or len(uvs) == 0 or any(vt is None for vt in faces_vt):
        return None

    # Crée l'image
    img = Image.new("RGB", (tex_w, tex_h), color=bg)
    draw = ImageDraw.Draw(img, "RGBA")

    # Rasterisation triangle par triangle
    # NB: On utilise une approximation "couleur moyenne par triangle".
    #     Pour un baking lisse barycentrique, il faudra implémenter
    #     une rasterisation scanline + interpolation des couleurs.
    scale = np.array([tex_w - 1, tex_h - 1], dtype=np.float32)

    for fv, fvt in zip(faces_v, faces_vt):
        if fvt is None:
            continue
        tri_uv = (uvs[np.array(fvt, dtype=int)] * scale).astype(np.float32)  # 3x2
        tri_col = vertex_rgb[np.array(fv, dtype=int)]                         # 3x3 (uint8)
        _rasterize_triangle(draw, tri_uv, tri_col)

    return img
