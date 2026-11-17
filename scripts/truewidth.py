"""True-thickness factor |u . N| for each composite (revision 2).

u = unit drilling direction in the unfolded frame (s, z, n), from the composite centres along the hole.
N = unit normal of the local lens plane n = a + b s + c z, fitted to the logged MS contacts within 60 m (in s, z);
    with fewer than 8 contacts there, within 150 m; otherwise the n axis (a lens parallel to the frame).
The factor converts down-hole length to thickness across the lens (t = L |u . N|) and is floored at 0.25,
so that a hole drilled nearly along a lens does not give a width of almost zero.
Replaces |dn/dl| (the horizontal across-frame component only), which overstates thickness for dipping lenses.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

FLOOR = 0.25


def drilling_direction(c):
    """c sorted by hole, f; with columns s, z, n, f, t. Returns an (n, 3) array of unit vectors."""
    out = np.full((len(c), 3), np.nan)
    mid = ((c.f + c.t) / 2).to_numpy()
    S = c[['s', 'z', 'n']].to_numpy()
    for h, idx in c.groupby('hole', sort=False).indices.items():
        if len(idx) < 2:
            continue
        m = mid[idx]
        g = np.column_stack([np.gradient(S[idx, k], m) for k in range(3)])
        out[idx] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
    return out


def lens_normals(P, contacts_szn, r1=60.0, r2=150.0, kmin=8):
    """Local lens-plane normals at points P (s, z) from contact points (s, z, n)."""
    tree = cKDTree(contacts_szn[:, :2]); N = np.tile([0.0, 0.0, 1.0], (len(P), 1)); src = np.zeros(len(P), int)
    for i, p in enumerate(P):
        for code, r in ((1, r1), (2, r2)):
            idx = tree.query_ball_point(p, r)
            if len(idx) >= kmin:
                Q = contacts_szn[idx]
                A = np.column_stack([np.ones(len(idx)), Q[:, 0] - p[0], Q[:, 1] - p[1]])
                co = np.linalg.lstsq(A, Q[:, 2], rcond=None)[0]
                v = np.array([-co[1], -co[2], 1.0]); N[i] = v / np.linalg.norm(v); src[i] = code
                break
    return N, src


def true_across(c, contacts_szn):
    """Thickness factor per composite of c (sorted by hole, f)."""
    u = drilling_direction(c)
    N, _ = lens_normals(c[['s', 'z']].to_numpy(), contacts_szn)
    f = np.abs((u * N).sum(1))
    f = np.where(np.isnan(f), 1.0, f)
    return np.clip(f, FLOOR, 1.0)
