# -------------------------------------------------------
# Visualisation 3D/2D du volume interpolé + points d'observation + maillage .OBJ
# -------------------------------------------------------
from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.ndimage import gaussian_filter, zoom

# Optionnels
try:
    from skimage.measure import marching_cubes
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import mplcursors
    _HAS_MPLCURSORS = True
except Exception:
    _HAS_MPLCURSORS = False

try:
    import trimesh
    _HAS_TRIMESH = True
except Exception:
    _HAS_TRIMESH = False


# ---------- Helpers ----------------------------------------------------------

def _ensure_xyz_order(xs, ys, zs, vol):
    """S’assure que le volume est (nx,ny,nz)."""
    sx, sy, sz = len(xs), len(ys), len(zs)
    if vol.shape == (sx, sy, sz):
        return vol
    if vol.shape == (sz, sy, sx):
        return np.transpose(vol, (2, 1, 0))
    raise ValueError(f"Shape {vol.shape} inattendue.")


def _slice_Z(vol, k): return vol[:, :, k].T


def _compute_isolevels(vol, n_isos, vmin, vmax):
    vv = vol[np.isfinite(vol)]
    if vmin is None: vmin = float(np.percentile(vv, 10))
    if vmax is None: vmax = float(np.percentile(vv, 90))
    levels = np.linspace(vmin, vmax, n_isos + 2)[1:-1]
    return levels, vmin, vmax


def _plot_isosurface(ax3d, xs, ys, zs, V, level, face_alpha=0.25,
                     cmap="viridis", mc_step=2, decimate_max_faces=60000,
                     fallback_scatter_step=3):
    denom = np.ptp(V) + 1e-12
    iso_color = plt.get_cmap(cmap)((level - np.nanmin(V)) / denom)
    if _HAS_SKIMAGE:
        vol_zyx = np.transpose(V, (2, 1, 0))
        dz, dy, dx = np.ptp(zs)/len(zs), np.ptp(ys)/len(ys), np.ptp(xs)/len(xs)
        verts, faces, _, _ = marching_cubes(vol_zyx, level=level,
                                            spacing=(dz, dy, dx),
                                            step_size=int(max(1, mc_step)))
        verts_xyz = np.stack([verts[:, 2]+xs[0], verts[:, 1]+ys[0], verts[:, 0]+zs[0]], 1)
        if decimate_max_faces and _HAS_TRIMESH:
            try:
                m = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)
                if len(m.faces) > decimate_max_faces:
                    m = m.simplify_quadratic_decimation(decimate_max_faces)
                verts_xyz, faces = m.vertices, m.faces
            except Exception:
                pass
        mesh = Poly3DCollection(verts_xyz[faces], alpha=face_alpha, linewidths=0)
        mesh.set_facecolor(iso_color)
        mesh.set_edgecolor("none")
        ax3d.add_collection3d(mesh)
    else:
        tol = (np.nanmax(V)-np.nanmin(V))*0.02
        mask = np.isfinite(V) & (np.abs(V - level) < tol)
        mask[::fallback_scatter_step, ::fallback_scatter_step, ::fallback_scatter_step] &= True
        xs3d, ys3d, zs3d = np.meshgrid(xs, ys, zs, indexing="xy")
        X, Y, Z = xs3d[mask], ys3d[mask], zs3d[mask]
        ax3d.scatter(X, Y, Z, s=1, alpha=0.15, c=[iso_color])


def _scatter_points_3d(ax, df, neon="#00FFFF"):
    x, y, z = df["X"], df["Y"], df["Z"]
    ax.scatter(x, y, z, s=24, c="k", alpha=1.0, depthshade=False, zorder=5)
    pts = ax.scatter(x, y, z, s=14, c=neon, alpha=1.0, depthshade=False, zorder=6)

    def _on_pick(event):
        i = event.ind[0]
        rtxt = f"R={df['R'].values[i]:.3g} " if "R" in df.columns else ""
        label = f"{rtxt}({x[i]:.2f},{y[i]:.2f},{z[i]:.2f})"
        ax.text(x[i], y[i], z[i], label, color="w",
                bbox=dict(fc="k", ec="none", alpha=0.6))
        ax.figure.canvas.draw_idle()

    pts.set_picker(True)
    ax.figure.canvas.mpl_connect("pick_event", _on_pick)
    return pts


def _scatter_points_2d(ax, df, neon="#00FFFF"):
    x, y = df["X"], df["Y"]
    ax.scatter(x, y, s=60, c="k", alpha=1.0, zorder=5)
    sc = ax.scatter(x, y, s=30, c=neon, alpha=1.0, zorder=6)
    if _HAS_MPLCURSORS:
        cursor = mplcursors.cursor(sc, hover=True)
        @cursor.connect("add")
        def _(sel):
            i = getattr(sel, "index", None)
            if i is None:
                xy = sc.get_offsets()
                d = np.sum((xy - sel.target)**2, 1)
                i = int(np.argmin(d))
            rtxt = f"R={df['R'].values[i]:.3g}\n" if "R" in df.columns else ""
            sel.annotation.set_text(f"{rtxt}X={x[i]:.2f}\nY={y[i]:.2f}")
            sel.annotation.get_bbox_patch().set(fc="k", ec="none", alpha=0.75)
    return sc


# ---------- Mesh helpers -----------------------------------------------------

def _load_mesh(path):
    m = trimesh.load_mesh(str(path), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.dump().values()))
    return m


def _mesh_edges_segments3d(mesh, max_edges=80000):
    F = np.asarray(mesh.faces, dtype=int)
    V = np.asarray(mesh.vertices, dtype=float)
    if F.size == 0:
        return np.empty((0, 2, 3), float)

    # Edges uniques
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)
    E = np.unique(E, axis=0)

    # Sous-échantillonnage si trop d'arêtes
    if len(E) > max_edges:
        # ➜ indices ENTIERs (pas float); pas d'ambiguïté sur les args de linspace
        idx = np.linspace(0, len(E) - 1, num=max_edges, endpoint=True, dtype=int)
        # sécurité: enlever doublons potentiels
        idx = np.unique(idx)
        E = E[idx]

    segs = np.stack([V[E[:, 0]], V[E[:, 1]]], axis=1)  # (n,2,3)
    return segs

def _mesh_section_segments_xy(mesh, z_value):
    sec3d = mesh.section(plane_origin=[0,0,float(z_value)], plane_normal=[0,0,1])
    if sec3d is None: return []
    path2d, _ = sec3d.to_planar()
    segs = []
    try:
        for P in path2d.discrete:
            P = np.asarray(P,float)
            if len(P)>=2: segs.append(P[:,:2])
    except Exception:
        V = np.asarray(path2d.vertices,float)
        for e in path2d.entities:
            if hasattr(e,"points"):
                idx = np.asarray(e.points,int); segs.append(V[idx])
    return segs


def _union_axes_limits(ax, xs, ys, zs, mesh=None):
    xmin,xmax=np.min(xs),np.max(xs)
    ymin,ymax=np.min(ys),np.max(ys)
    zmin,zmax=np.min(zs),np.max(zs)
    if mesh is not None:
        V=np.asarray(mesh.vertices,float)
        xmin=min(xmin,V[:,0].min()); xmax=max(xmax,V[:,0].max())
        ymin=min(ymin,V[:,1].min()); ymax=max(ymax,V[:,1].max())
        zmin=min(zmin,V[:,2].min()); zmax=max(zmax,V[:,2].max())
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_zlim(zmin,zmax)


def _warn_if_far(ax, xs, ys, zs, mesh):
    gv=np.array([np.mean(xs),np.mean(ys),np.mean(zs)])
    mv=np.mean(mesh.vertices,0)
    d=np.linalg.norm(gv-mv)
    scale=max(np.ptp(xs),np.ptp(ys),np.ptp(zs))
    if d>0.1*scale:
        ax.text2D(0.02,0.98,f"⚠ Mesh/volume décalés (Δ≈{d:.2f})",
                  transform=ax.transAxes,color="red",
                  bbox=dict(fc="w",ec="none",alpha=0.8))

def _add_mesh_wire3d(ax3d, mesh, color="#aaaaaa", alpha=0.8, lw=0.5, max_edges=80000):
    """
    Ajoute un wireframe léger sous forme de Line3DCollection.
    Requiert: from mpl_toolkits.mplot3d.art3d import Line3DCollection
    """
    segs = _mesh_edges_segments3d(mesh, max_edges=max_edges)
    if segs.size == 0:
        return None
    coll = Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha)
    ax3d.add_collection3d(coll)
    return coll



# ---------- Main viewer ------------------------------------------------------

def show_volume_and_points(
    xs, ys, zs, volume, points_df=None,
    n_isosurfaces=3, vmin=None, vmax=None, cmap="viridis",
    fast=True, downsample_factor=0.5,
    mc_step=2, decimate_max_faces=60000, fallback_scatter_step=3,
    obj_path=None, mesh_mode="wire", mesh_color="#a0a0a0",
    mesh_alpha=0.9, mesh_lw=0.5, mesh_max_edges=80000,
    show_isosurfaces=True, show_section_2d=True, title="Volume & Points"
):
    """Affichage interactif 3D+2D avec maillage .obj léger."""
    V=_ensure_xyz_order(xs,ys,zs,volume)
    if fast and downsample_factor!=1.0:
        V=zoom(V,downsample_factor,order=1)
        xs=np.linspace(xs.min(),xs.max(),V.shape[0])
        ys=np.linspace(ys.min(),ys.max(),V.shape[1])
        zs=np.linspace(zs.min(),zs.max(),V.shape[2])
    nx,ny,nz=V.shape
    levels,vmin,vmax=_compute_isolevels(V,n_isosurfaces,vmin,vmax)

    fig=plt.figure(figsize=(13,6)); fig.suptitle(title)
    ax3d=fig.add_subplot(1,2,1,projection="3d")
    ax2d=fig.add_subplot(1,2,2)

    mesh=None
    if obj_path and _HAS_TRIMESH:
        try:
            mesh=_load_mesh(obj_path)
        except Exception as e:
            print(f"[WARN] OBJ non chargé: {e}")

    # --- 3D
    if show_isosurfaces and mesh is None:
        for lv in levels:
            _plot_isosurface(ax3d,xs,ys,zs,V,lv,0.25,cmap,mc_step,decimate_max_faces,fallback_scatter_step)
    if mesh is not None:
        if mesh_mode=="wire":
            _add_mesh_wire3d(ax3d,mesh,color=mesh_color,alpha=mesh_alpha,lw=mesh_lw,max_edges=mesh_max_edges)
        else:
            Vt=np.asarray(mesh.vertices); F=np.asarray(mesh.faces,int)
            coll=Poly3DCollection(Vt[F],alpha=0.15,facecolor=mesh_color,edgecolor="none")
            ax3d.add_collection3d(coll)
        _union_axes_limits(ax3d,xs,ys,zs,mesh)
        _warn_if_far(ax3d,xs,ys,zs,mesh)
    else:
        _union_axes_limits(ax3d,xs,ys,zs)
    if points_df is not None: _scatter_points_3d(ax3d,points_df)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    # --- 2D
    k0=nz//2
    sl=_slice_Z(V,k0)
    sl=zoom(sl,2.0,order=1); sl=gaussian_filter(sl,0.8)
    im=ax2d.imshow(sl,origin="lower", extent=(xs.min(),xs.max(),ys.min(),ys.max()), vmin=vmin,vmax=vmax,cmap=cmap)
    plt.colorbar(im,ax=ax2d, fraction=0.046, pad=0.04).set_label("Radioactivity (cpm)")
    if points_df is not None: _scatter_points_2d(ax2d,points_df)
    ax2d.set_title(f"Coupe Z @ {zs[k0]:.2f}"); ax2d.set_xlabel("X"); ax2d.set_ylabel("Y")

    sec=None
    if mesh is not None and show_section_2d:
        segs=_mesh_section_segments_xy(mesh,zs[k0])
        if segs:
            sec=LineCollection(segs,colors="w",lw=1.5,alpha=0.9)
            ax2d.add_collection(sec)

    # --- slider
    axsl=plt.axes([0.56,0.08,0.35,0.03])
    slz=Slider(axsl,"Z index",0,nz-1,valinit=k0,valfmt="%0.0f")
    def _upd(val):
        k=int(slz.val)
        img=_slice_Z(V,k)
        img=zoom(img,2.0,order=1); img=gaussian_filter(img,0.8)
        im.set_data(img)
        ax2d.set_title(f"Coupe Z @ {zs[k]:.2f}")
        if mesh is not None and show_section_2d:
            segs=_mesh_section_segments_xy(mesh,zs[k])
            if sec: sec.set_segments(segs)
        fig.canvas.draw_idle()
    slz.on_changed(_upd)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        plt.tight_layout(rect=[0,0.05,1,0.97])
    plt.show()
