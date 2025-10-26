from __future__ import annotations
from pathlib import Path
import numpy as np


class SimpleOBJ:
    """
    Représentation minimale d'un fichier OBJ pour nos besoins :
      - vertices : (N,3) float64
      - uvs      : (M,2) float64 (éventuellement vide)
      - faces_v  : list[[i,j,k] ...] indices de sommets (0-based)
      - faces_vt : list[[iu,iv,iw] ...] indices UV (0-based) ou None si absents
      - mtllib   : nom du fichier .mtl référencé (str ou None)
      - usemtl   : nom du matériau courant (str ou None)

    Notes :
      - On ne gère pas ici les normales (vn) car elles ne sont pas nécessaires
        pour colorer le maillage ; on peut les ajouter si besoin.
      - Les faces N-gones sont supposées triangulées. Si ce n'est pas le cas,
        l'écriture restera correcte mais le baking simple suppose des triangles.
    """

    def __init__(self):
        self.vertices: np.ndarray | list = []
        self.uvs: np.ndarray | list = []
        self.faces_v: list[list[int]] = []
        self.faces_vt: list[list[int] | None] = []
        self.mtllib: str | None = None
        self.usemtl: str | None = None

    # ---------------------------
    # Helpers parsing indices f
    # ---------------------------
    @staticmethod
    def _parse_face_token(tok: str) -> tuple[int, int | None]:
        """
        token typique : 'v', 'v/vt', 'v//vn', 'v/vt/vn'
        Retourne : (v_idx0, vt_idx0_or_None), indices 0-based
        """
        parts = tok.split("/")
        v = int(parts[0]) - 1
        vt = int(parts[1]) - 1 if len(parts) > 1 and parts[1] != "" else None
        return v, vt

    # ---------------------------
    # Load
    # ---------------------------
    @classmethod
    def load(cls, path: str | Path) -> "SimpleOBJ":
        obj = cls()
        path = Path(path)
        verts = []
        uvs = []
        faces_v = []
        faces_vt = []
        mtllib = None
        usemtl = None

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                if line.startswith("mtllib "):
                    mtllib = line.strip().split(maxsplit=1)[1]
                elif line.startswith("usemtl "):
                    usemtl = line.strip().split(maxsplit=1)[1]
                elif line.startswith("v "):
                    # ATTENTION : certains OBJ incluent des couleurs après z ; on ignore à la lecture.
                    parts = line.strip().split()
                    if len(parts) < 4:
                        continue
                    _, x, y, z = parts[:4]
                    verts.append([float(x), float(y), float(z)])
                elif line.startswith("vt "):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        _, u, v = parts[:3]
                        uvs.append([float(u), float(v)])
                elif line.startswith("f "):
                    toks = line.strip().split()[1:]
                    v_idx = []
                    vt_idx = []
                    has_vt = True
                    for t in toks:
                        vi, vti = cls._parse_face_token(t)
                        v_idx.append(vi)
                        vt_idx.append(vti)
                        if vti is None:
                            has_vt = False
                    faces_v.append(v_idx)
                    faces_vt.append(vt_idx if has_vt else None)

        obj.vertices = np.asarray(verts, dtype=float) if verts else np.empty((0, 3), dtype=float)
        obj.uvs = np.asarray(uvs, dtype=float) if uvs else np.empty((0, 2), dtype=float)
        obj.faces_v = faces_v
        obj.faces_vt = faces_vt
        obj.mtllib = mtllib
        obj.usemtl = usemtl
        return obj

    # ---------------------------
    # Save with vertex colors
    # ---------------------------
    def save_with_vertex_colors(self, path_out: str | Path, rgb: np.ndarray) -> None:
        """
        Écrit un OBJ incluant des couleurs par sommet :
          v x y z r g b   (où r,g,b ∈ [0..1])

        De nombreux outils (MeshLab, Blender, Trimesh) savent lire ce format
        'dé facto', bien que non normé dans la spécification historique d'OBJ.

        Paramètres
        ----------
        path_out : str | Path
            Chemin de sortie de l'OBJ.
        rgb : np.ndarray
            Tableau (N,3) uint8 ou float indiquant la couleur par sommet.
        """
        path_out = Path(path_out)

        if isinstance(rgb, np.ndarray) and rgb.dtype != float:
            rgb01 = (rgb.astype(float) / 255.0)
        else:
            rgb01 = np.clip(rgb, 0.0, 1.0)

        if len(rgb01) != len(self.vertices):
            raise ValueError(
                f"Nombre de couleurs ({len(rgb01)}) != nombre de sommets ({len(self.vertices)})"
            )

        with path_out.open("w", encoding="utf-8") as f:
            # mtllib / usemtl si présents
            if self.mtllib:
                f.write(f"mtllib {self.mtllib}\n")
            if self.usemtl:
                f.write(f"usemtl {self.usemtl}\n")

            # Sommets avec couleurs
            for (x, y, z), (r, g, b) in zip(self.vertices, rgb01):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

            # UV (si existants)
            for uv in self.uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Faces : on conserve le mapping v/vt si présent
            for fv, fvt in zip(self.faces_v, self.faces_vt):
                if fvt is not None:
                    f.write("f " + " ".join(f"{vi+1}/{ti+1}" for vi, ti in zip(fv, fvt)) + "\n")
                else:
                    f.write("f " + " ".join(f"{vi+1}" for vi in fv) + "\n")
