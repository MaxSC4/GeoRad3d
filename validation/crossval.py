from __future__ import annotations

import numpy as np
from pykrige.ok3d import OrdinaryKriging3D

try:  # joblib permet de paralléliser la CV si dispo
    from joblib import Parallel, delayed
except Exception:  # pragma: no cover - dépend de l'environnement utilisateur
    Parallel = delayed = None


def _loo_single(
    idx: int,
    xyz: np.ndarray,
    rvals: np.ndarray,
    variogram_model: str,
    variogram_parameters,
) -> tuple[float, float]:
    """Calcule la prédiction LOO pour un index donné."""
    n = len(rvals)
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    ok = OrdinaryKriging3D(
        xyz[mask, 0], xyz[mask, 1], xyz[mask, 2], rvals[mask],
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        enable_plotting=False, verbose=False
    )
    est, v = ok.execute("points", xyz[idx:idx+1, 0], xyz[idx:idx+1, 1], xyz[idx:idx+1, 2])
    return float(est), float(v)


def loo_metrics(
    xyz,
    rvals,
    variogram_model: str = "spherical",
    variogram_parameters=None,
    n_jobs: int | None = None,
):
    """
    Leave-One-Out: pour chaque i, krige sur N-1 points et prédit au point i.
    Renvoie dict métriques + arrays (pred, var)
    """
    xyz = np.asarray(xyz, dtype=float)
    rvals = np.asarray(rvals, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz doit être de shape (N,3)")

    n = len(rvals)
    pred = np.full(n, np.nan, dtype=float)
    var = np.full(n, np.nan, dtype=float)

    # n_jobs = 1 ➜ séquentiel, n_jobs=None/-1 ➜ auto (si joblib dispo).
    if n_jobs == 0:
        n_jobs = 1
    parallel_supported = Parallel is not None and (n_jobs is None or n_jobs != 1) and n >= 4

    if parallel_supported:
        effective_jobs = -1 if n_jobs is None else n_jobs
        worker = delayed(_loo_single)
        results = Parallel(n_jobs=effective_jobs, prefer="processes")(
            worker(i, xyz, rvals, variogram_model, variogram_parameters)
            for i in range(n)
        )
        preds, variances = zip(*results)
        pred[:] = np.asarray(preds, dtype=float)
        var[:] = np.asarray(variances, dtype=float)
    else:
        for i in range(n):
            est, v = _loo_single(i, xyz, rvals, variogram_model, variogram_parameters)
            pred[i] = est
            var[i] = v

    resid = rvals - pred
    me = np.nanmean(resid)
    rmse = float(np.sqrt(np.nanmean(resid ** 2)))
    # MSSE: mean standardized squared error ~1 si variance bien calibrée
    msse = float(np.nanmean((resid ** 2) / var))
    # VSE: variance of standardized errors ~1 si ok
    z = resid / np.sqrt(var)
    vse = float(np.nanvar(z, ddof=1))

    return {
        "ME": float(me), "RMSE": rmse, "MSSE": msse, "VSE": vse,
        "pred": pred, "var": var
    }
