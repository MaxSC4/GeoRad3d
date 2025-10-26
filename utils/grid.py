import numpy as np
from typing import Tuple

def regular_grid(bx, by, bz, nx=60, ny=60, nz=40):
    xs = np.linspace(bx[0], bx[1], nx)
    ys = np.linspace(by[0], by[1], ny)
    zs = np.linspace(bz[0], bz[1], nz)
    return xs, ys, zs
