from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class StratumDefinition:
    name: str
    origin: tuple[float, float, float]
    dip_deg: float
    azimuth_deg: float
    value: float
    thickness: float = 0.0
    spacing: float = 8.0
    n_layers: int | None = None


def _coerce_float_triplet(val) -> tuple[float, float, float]:
    if not isinstance(val, Iterable):
        raise ValueError("origin must be an iterable of length 3")
    vals = list(val)
    if len(vals) != 3:
        raise ValueError("origin must contain exactly three values")
    return tuple(float(v) for v in vals)


def load_strata_file(path: str | Path) -> list[StratumDefinition]:
    """
    Lit un fichier JSON décrivant des strates inclinées.

    Format attendu :
    {
      "strata": [
        {
          "name": "Layer A",
          "origin": [x0, y0, z0],
          "dip_deg": 18,
          "azimuth_deg": 90,
          "thickness": 5,
          "spacing": 10,
          "value": 150,
          "n_layers": 3
        }
      ]
    }
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    strata_list = data["strata"] if isinstance(data, dict) and "strata" in data else data
    parsed: list[StratumDefinition] = []
    for i, raw in enumerate(strata_list):
        name = str(raw.get("name", f"stratum_{i}"))
        origin = _coerce_float_triplet(raw.get("origin", (0.0, 0.0, 0.0)))
        dip = float(raw.get("dip_deg", 0.0))
        azimuth = float(raw.get("azimuth_deg", 0.0))
        value = float(raw["value"])
        thickness = float(raw.get("thickness", 0.0))
        spacing = float(raw.get("spacing", 8.0))
        n_layers = raw.get("n_layers")
        if n_layers is not None:
            n_layers = max(1, int(n_layers))
        parsed.append(
            StratumDefinition(
                name=name,
                origin=origin,
                dip_deg=dip,
                azimuth_deg=azimuth,
                value=value,
                thickness=abs(thickness),
                spacing=max(0.5, spacing),
                n_layers=n_layers,
            )
        )
    return parsed


def _layer_offsets(stratum: StratumDefinition) -> np.ndarray:
    if stratum.n_layers is not None:
        n = max(1, int(stratum.n_layers))
    else:
        if stratum.thickness <= 0:
            return np.array([0.0], dtype=float)
        n = int(np.clip(np.round(stratum.thickness / max(1.0, stratum.spacing)) + 1, 2, 12))
    if n == 1 or stratum.thickness <= 0:
        return np.array([0.0], dtype=float)
    return np.linspace(-stratum.thickness / 2, stratum.thickness / 2, n, dtype=float)


def _plane_z(stratum: StratumDefinition, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    x0, y0, z0 = stratum.origin
    dx = xs - x0
    dy = ys - y0
    dip_rad = np.deg2rad(stratum.dip_deg)
    az_rad = np.deg2rad(stratum.azimuth_deg)
    along = np.cos(az_rad) * dx + np.sin(az_rad) * dy
    dz = np.tan(dip_rad) * along
    return z0 + dz


def sample_strata_points(
    strata: list[StratumDefinition],
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    global_spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Crée des pseudo-points (XYZ + valeur) décrivant les strates inclinées.
    Ils sont utilisés comme conditions initiales dans le krigeage.
    """
    bx, by, bz = bbox
    xs_all: list[np.ndarray] = []
    vals_all: list[np.ndarray] = []

    for stratum in strata:
        spacing = float(global_spacing or stratum.spacing)
        xs = np.arange(bx[0], bx[1] + spacing * 0.5, spacing)
        ys = np.arange(by[0], by[1] + spacing * 0.5, spacing)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        base_z = _plane_z(stratum, grid_x, grid_y)
        offsets = _layer_offsets(stratum)
        for offset in offsets:
            layer_z = base_z + offset
            mask = (layer_z >= bz[0] - spacing) & (layer_z <= bz[1] + spacing)
            if not np.any(mask):
                continue
            coords = np.column_stack([grid_x[mask], grid_y[mask], layer_z[mask]])
            xs_all.append(coords)
            vals_all.append(np.full(len(coords), stratum.value, dtype=float))

    if not xs_all:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    return np.vstack(xs_all), np.concatenate(vals_all)
