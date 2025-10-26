import numpy as np
from pykrige.ok3d import OrdinaryKriging3D

def loo_metrics(xyz, rvals, variogram_model="spherical", variogram_parameters=None):
    """
    Leave-One-Out: pour chaque i, krige sur N-1 points et prédit au point i.
    Renvoie dict métriques + arrays (pred, var)
    """
    n = len(rvals)
    pred = np.full(n, np.nan)
    var  = np.full(n, np.nan)

    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        ok = OrdinaryKriging3D(
            xyz[mask,0], xyz[mask,1], xyz[mask,2], rvals[mask],
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            enable_plotting=False, verbose=False
        )
        est, v = ok.execute("points", xyz[i:i+1,0], xyz[i:i+1,1], xyz[i:i+1,2])
        pred[i] = float(est)
        var[i]  = float(v)

    resid = rvals - pred
    me   = np.nanmean(resid)
    rmse = float(np.sqrt(np.nanmean(resid**2)))
    # MSSE: mean standardized squared error ~1 si variance bien calibrée
    msse = float(np.nanmean((resid**2) / var))
    # VSE: variance of standardized errors ~1 si ok
    z = resid / np.sqrt(var)
    vse = float(np.nanvar(z, ddof=1))

    return {
        "ME": float(me), "RMSE": rmse, "MSSE": msse, "VSE": vse,
        "pred": pred, "var": var
    }
