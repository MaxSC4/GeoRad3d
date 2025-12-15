import numpy as np

def regular_grid(bx, by, bz, nx=60, ny=60, nz=40):
    """Retourne trois axes réguliers (X, Y, Z) couvrant les bornes fournies."""
    xs = np.linspace(bx[0], bx[1], nx)
    ys = np.linspace(by[0], by[1], ny)
    zs = np.linspace(bz[0], bz[1], nz)
    return xs, ys, zs
