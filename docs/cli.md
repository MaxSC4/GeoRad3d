# GeoRad3D — Command Line Interface (CLI)

This document describes the **main commands** available in `main.py`, their **arguments**, and **example usages**.
Each command represents a key step in the GeoRad3D pipeline — from data interpolation to visualization and 3D surface projection.

---

## Command: `fit`

### Description
Interpolates the radioactivity in 3D from point measurements `(X, Y, Z, R)` using **kriging** (`pykrige`)
and saves the resulting 3D volume (`.npz`) in `outputs/volumes/`.

Optionally performs a **Leave-One-Out cross-validation (LOO)** and saves diagnostic metrics.

### Syntax
```
python main.py fit [options]
```

### Options
| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--csv` | str | `data/raw/points.csv` | Path to CSV file containing `(X, Y, Z, R)` points |
| `--pad` | float | 0.05 | Relative margin added to the bounding box before interpolation |
| `--nx`, `--ny`, `--nz` | int | 60 / 60 / 40 | Grid resolution in each dimension |
| `--variogram` | str | `"spherical"` | Variogram model (`spherical`, `exponential`, `gaussian`, `linear`) |
| `--preview` | flag |  | Open a quick visualization after fitting |
| `--skip-cv` | flag |  | Skip cross-validation (faster) |

### Example
```
python main.py fit --csv data/raw/points.csv --nx 80 --ny 80 --nz 50 --variogram gaussian --skip-cv
```

---

## Command: `view`

### Description
Displays an **interactive 3D visualization** of the interpolated volume and measurement points,  
along with a **dynamic 2D slice view** (controlled by a Z-slider).

Optionally overlays a geological model (`.obj`) for contextual visualization.

### Syntax
```
python main.py view [options]
```

### Options
| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--csv` | str | `data/raw/points.csv` | CSV file with measurement points |
| `--nisos` | int | 3 | Number of isosurfaces to display |
| `--vmin`, `--vmax` | float | auto | Color scale bounds |
| `--cmap` | str | `"viridis"` | Matplotlib colormap |
| `--obj` | str | *None* | Path to a `.obj` model to overlay |
| `--no-isos` | flag |  | Disable isosurface rendering (faster) |
| `--mesh-mode` | str | `"wire"` | Mesh display mode: `wire` or `solid` |
| `--mesh-max-edges` | int | 80000 | Limit number of wireframe segments for performance |

### Examples
```
python main.py view --csv data/raw/points.csv --nisos 4 --cmap plasma
```

```
python main.py view --csv data/raw/points.csv --obj data/meshes/model_georef.obj --no-isos --mesh-mode wire --mesh-max-edges 40000
```

---

## Command: `paint`

### 🔹 Description
Projects interpolated values from the 3D volume onto the surface of a `.obj` mesh.  
Generates a new `.obj` with **per-vertex color information** and optionally a **baked texture (.png/.mtl)**.

### Syntax
```
python main.py paint [options]
```

### Options
| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--obj` | str | `data/meshes/affleurement.obj` | Path to the mesh to paint |
| `--out-dir` | str | `outputs/meshes/` | Output directory |
| `--cmap` | str | `"viridis"` | Colormap for value mapping |
| `--vmin`, `--vmax` | float | auto | Value range to map |
| `--no-texture` | flag |  | Disable texture/MTL export (vertex colors only) |
| `--tex-w`, `--tex-h` | int | 2048 / 2048 | Texture resolution |
| `--no-nan-color` | flag |  | Skip recoloring of vertices outside the interpolated volume |

### Example
```
python main.py paint --obj data/meshes/model_georef.obj --cmap inferno --tex-w 4096 --tex-h 4096
```

---

## Command: `gif`

### Description
Generates an animated `.gif` showing the **evolution of the 2D slice** through the interpolated volume along the Z axis,  
with synchronized slider motion and optional point overlay.

### Syntax
```
python main.py gif [options]
```

### ⚙️ Options
| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--csv` | str | `data/raw/points.csv` | Points to overlay on the slice |
| `--out` | str | `outputs/figures/z_sweep.gif` | Output path for the GIF |
| `--cmap` | str | `"viridis"` | Colormap |
| `--vmin`, `--vmax` | float | auto | Color scale bounds |
| `--upsample` | float | 2.0 | Upsampling factor for smoother frames |
| `--sigma` | float | 0.8 | Gaussian blur factor |
| `--skip` | int | 1 | Slice stride (1 = every slice) |
| `--fps` | int | 10 | Frames per second |
| `--dpi` | int | 120 | Output resolution |

### Example
```
python main.py gif --csv data/raw/points.csv --out outputs/figures/z_sweep.gif --cmap plasma --fps 8
```

---

## Command: `align-obj` *(coming soon)*

Planned for automating `.obj` **georeferencing** via the  
[`align_obj_to_points`](../utils/obj_io.py) function — applying translation, scaling, and rotation  
to align local mesh coordinates with measured 3D points.

---

## Summary

| Command | Purpose | Output |
|----------|----------|--------|
| `fit` | 3D interpolation + cross-validation | `outputs/volumes/volume.npz` |
| `view` | Interactive 3D/2D visualization | Matplotlib window |
| `paint` | Project interpolated values on mesh | Colored `.obj` / `.png` texture |
| `gif` | Animated 2D Z-slice | `.gif` |
| `align-obj` *(soon)* | Align local .obj to geospatial data | `.obj` aligned model |
