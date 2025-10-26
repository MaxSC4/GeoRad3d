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

    def save(self, path_out: str | Path) -> None:
        """
        Écrit un OBJ "classique" (sans couleurs par sommet).
        On préserve mtllib/usemtl, les UV (vt) et les indices de faces (v/vt).
        """
        path_out = Path(path_out)
        with path_out.open("w", encoding="utf-8") as f:
            if self.mtllib:
                f.write(f"mtllib {self.mtllib}\n")
            if self.usemtl:
                f.write(f"usemtl {self.usemtl}\n")

            for x, y, z in np.asarray(self.vertices, dtype=float):
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

            for uv in np.asarray(self.uvs, dtype=float):
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            for fv, fvt in zip(self.faces_v, self.faces_vt):
                if fvt is not None:
                    f.write("f " + " ".join(f"{vi+1}/{ti+1}" for vi, ti in zip(fv, fvt)) + "\n")
                else:
                    f.write("f " + " ".join(f"{vi+1}" for vi in fv) + "\n")



# ---------------------------
# Helpers d'alignement
# ---------------------------
def _load_points_any(points_source):
    """
    Accepte: chemin CSV (X,Y,Z), pandas.DataFrame(X,Y,Z), ou np.ndarray (N,3).
    Retourne np.ndarray float (N,3).
    """
    try:
        import pandas as pd  # lazy import
    except Exception:
        pd = None

    if isinstance(points_source, (str, Path)):
        if pd is None:
            raise ImportError("pandas est requis pour lire un CSV.")
        df = pd.read_csv(points_source)
        return df[["X", "Y", "Z"]].to_numpy(dtype=float)

    if pd is not None:
        try:
            import pandas as pd  # noqa
            if isinstance(points_source, pd.DataFrame):
                return points_source[["X", "Y", "Z"]].to_numpy(dtype=float)
        except Exception:
            pass

    arr = np.asarray(points_source, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points_source doit être (N,3) en X,Y,Z.")
    return arr


def _scale_about_center(vertices: np.ndarray, center: np.ndarray, scale_xyz: tuple[float, float, float]):
    shifted = vertices - center[None, :]
    scaled = shifted * np.array(scale_xyz, dtype=float)[None, :]
    return scaled + center[None, :]


def _rotate_about_z(vertices: np.ndarray, center: np.ndarray, degrees: float):
    if abs(degrees) < 1e-12:
        return vertices
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0],
                  [s,  c, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    return (vertices - center[None, :]) @ R.T + center[None, :]


def align_obj_to_points(
    obj_in: str | Path,
    points_source,                 # CSV/DataFrame/ndarray (X,Y, Z)
    out_path: str | Path | None = None,
    mode: str = "translate",       # "translate" | "translate+scale"
    scale_axes: str = "xy",        # axes à scaler si mode="translate+scale"
    lock_z_translation: bool = False,  # True = aligne seulement en XY
    rotate_z_deg: float = 0.0,     # petite rotation autour de Z si besoin
) -> dict:
    """
    Aligne un .OBJ (repère local) sur des points géoréférencés (X,Y,Z).

    Étapes:
      1) calcule le centre du mesh et des points
      2) (option) mise à l'échelle par rapport aux étendues (axes choisis)
      3) (option) petite rotation autour de Z
      4) translation pour superposer les centres (XY ou XYZ)

    Retourne un dict: {'offset': (dx,dy,dz), 'scale': (sx,sy,sz), 'rotate_z_deg': ..., 'obj_out': "..."}
    """
    obj_in = Path(obj_in)
    obj = SimpleOBJ.load(obj_in)

    pts = _load_points_any(points_source)

    # Centres
    mesh_center = np.asarray(obj.vertices, dtype=float).mean(axis=0) if len(obj.vertices) else np.zeros(3)
    pts_center  = pts.mean(axis=0)

    # Mise à l'échelle grossière (sur étendue)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)
    if mode == "translate+scale":
        m_min, m_max = np.min(obj.vertices, axis=0), np.max(obj.vertices, axis=0)
        p_min, p_max = np.min(pts, axis=0), np.max(pts, axis=0)
        m_span = np.maximum(m_max - m_min, 1e-9)
        p_span = np.maximum(p_max - p_min, 1e-9)
        if "x" in scale_axes: scale[0] = float(p_span[0] / m_span[0])
        if "y" in scale_axes: scale[1] = float(p_span[1] / m_span[1])
        if "z" in scale_axes: scale[2] = float(p_span[2] / m_span[2])
        obj.vertices = _scale_about_center(np.asarray(obj.vertices, float), mesh_center, tuple(scale))

    # Rotation Z optionnelle (autour du centre AVANT translation)
    if abs(rotate_z_deg) > 1e-9:
        obj.vertices = _rotate_about_z(np.asarray(obj.vertices, float), mesh_center, rotate_z_deg)

    # Translation (XY ou XYZ)
    offset = pts_center - mesh_center
    if lock_z_translation:
        offset[2] = 0.0
    obj.vertices = np.asarray(obj.vertices, float) + offset[None, :]

    # Écriture
    if out_path is None:
        out_path = obj_in.with_name(obj_in.stem + "_aligned.obj")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obj.save(out_path)

    return {
        "offset": (float(offset[0]), float(offset[1]), float(offset[2])),
        "scale": (float(scale[0]), float(scale[1]), float(scale[2])),
        "rotate_z_deg": float(rotate_z_deg),
        "obj_out": str(out_path),
    }
