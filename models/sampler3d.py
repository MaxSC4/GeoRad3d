from __future__ import annotations
import numpy as np
from scipy.interpolate import RegularGridInterpolator


class ScalarFieldSampler:
    """
    Classe utilitaire pour échantillonner un champ scalaire
    défini sur une grille 3D régulière (volume interpolé).

    Exemple d’usage :
    >>> sampler = ScalarFieldSampler(xs, ys, zs, volume)
    >>> values = sampler.sample_xyz(np.array([[100.2, 50.1, 12.3],
                                              [105.7, 51.2, 14.5]]))
    """

    def __init__(self, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, values: np.ndarray):
        """
        Initialise l’interpolateur trilineaire à partir d’un volume régulier.

        Paramètres
        ----------
        xs, ys, zs : np.ndarray
            Coordonnées 1D de la grille (axes X, Y, Z, croissantes).
        values : np.ndarray
            Tableau 3D contenant les valeurs scalaires du champ.
            Peut être de shape (len(xs), len(ys), len(zs)) ou (len(zs), len(ys), len(xs)).
        """

        # Détection automatique de l’ordre du tableau
        if values.shape == (len(xs), len(ys), len(zs)):
            # Réordonne pour (Z, Y, X)
            V = np.transpose(values, (2, 1, 0))
        elif values.shape == (len(zs), len(ys), len(xs)):
            V = values
        else:
            raise ValueError(
                f"Shape du volume incohérente : {values.shape} "
                f"vs ({len(xs)}, {len(ys)}, {len(zs)}) ou ({len(zs)}, {len(ys)}, {len(xs)})"
            )

        self.xs = np.asarray(xs)
        self.ys = np.asarray(ys)
        self.zs = np.asarray(zs)
        self.V = V

        # Interpolateur SciPy : sur axes (Z, Y, X)
        self._interp = RegularGridInterpolator(
            (self.zs, self.ys, self.xs),
            self.V,
            bounds_error=False,
            fill_value=np.nan,
        )

    def sample_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """
        Interpole les valeurs du champ en des positions XYZ arbitraires.

        Paramètres
        ----------
        xyz : np.ndarray
            Tableau (N,3) des coordonnées réelles (X,Y,Z).

        Retour
        ------
        np.ndarray
            Tableau (N,) des valeurs interpolées (NaN si en dehors de la grille).
        """
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz doit être de shape (N, 3)")

        # RegularGridInterpolator attend (z, y, x)
        pts_zyx = np.stack([xyz[:, 2], xyz[:, 1], xyz[:, 0]], axis=1)
        return self._interp(pts_zyx)
