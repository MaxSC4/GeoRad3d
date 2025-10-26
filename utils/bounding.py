# -------------------------------------------------------
# Calcul des bornes X, Y, Z à partir des points
# avec options de padding asymétrique
# -------------------------------------------------------
import numpy as np

def bbox_from_points(x, y, z, pad_frac=0.05, pad_z_down=0.5, pad_z_up=0.0):
    """
    Calcule une bounding box (X,Y,Z) avec marge.
    pad_frac : fraction de marge horizontale (X/Y)
    pad_z_down : fraction de la hauteur ajoutée en dessous (profondeur)
    pad_z_up : fraction de la hauteur ajoutée au-dessus (air)
    """
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    z_min, z_max = np.nanmin(z), np.nanmax(z)

    dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min

    # Marges horizontales symétriques
    x_min -= dx * pad_frac
    x_max += dx * pad_frac
    y_min -= dy * pad_frac
    y_max += dy * pad_frac

    # Marges verticales asymétriques (z vers le bas)
    z_min -= dz * pad_z_down
    z_max += dz * pad_z_up

    return (x_min, x_max), (y_min, y_max), (z_min, z_max)
