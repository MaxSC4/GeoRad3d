import numpy as np
from pykrige.ok3d import OrdinaryKriging3D

def krige_volume(xs, ys, zs, xyz, rvals, variogram_model="spherical",
                 variogram_parameters=None, enable_plotting=False):
    """
    xyz: (N,3) points d'échantillonnage
    rvals: (N,) valeurs R
    Retourne volume de shape (len(xs), len(ys), len(zs)) (X,Y,Z)
    """
    ok3d = OrdinaryKriging3D(
        xyz[:,0], xyz[:,1], xyz[:,2], rvals,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        enable_plotting=enable_plotting,
        verbose=False
    )
    # PyKrige prend les grilles 1D; renvoie z est. + variance
    est, var = ok3d.execute("grid", xs, ys, zs)
    # est a typiquement shape (nx, ny, nz)
    return est, var
