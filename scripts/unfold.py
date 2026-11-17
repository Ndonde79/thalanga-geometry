"""Unfolded frame for the curved, near-vertical Thalanga ore horizon.

The plan trace of the ore horizon is a smoothed centreline Y(X) through the median northing of OH composites in
50 m easting bins. Each point maps to (s, z, n): s = distance along the centreline, z = elevation, n = signed
horizontal distance from the centreline (positive north). For lenses dipping 70-90 degrees this flattens the curve
so that one set of anisotropy ranges (along s, along z, across n) applies everywhere.
"""
import numpy as np
import pandas as pd


class Unfold:
    def __init__(self, comp: pd.DataFrame, bin_m: float = 50.0, smooth_bins: int = 5):
        oh = comp[comp.f_OH >= 0.5]
        x0, x1 = np.floor(oh.X.min() / bin_m) * bin_m, np.ceil(oh.X.max() / bin_m) * bin_m
        edges = np.arange(x0, x1 + bin_m, bin_m)
        mid = (edges[:-1] + edges[1:]) / 2
        yb = oh.groupby(pd.cut(oh.X, edges), observed=False).Y.median().to_numpy()
        ok = np.isfinite(yb)
        y = np.interp(mid, mid[ok], yb[ok])
        k = np.ones(smooth_bins) / smooth_bins
        ys = np.convolve(np.pad(y, smooth_bins // 2, mode='edge'), k, mode='valid')
        # densify the centreline to 5 m spacing
        xd = np.arange(mid[0], mid[-1] + 1e-9, 5.0)
        self.cx, self.cy = xd, np.interp(xd, mid, ys)
        seg = np.hypot(np.diff(self.cx), np.diff(self.cy))
        self.cs = np.concatenate([[0.0], np.cumsum(seg)])

    def __call__(self, X):
        X = np.asarray(X, float)
        from scipy.spatial import cKDTree
        P = np.column_stack([self.cx, self.cy])
        _, i = cKDTree(P).query(X[:, :2])
        i = np.clip(i, 0, len(P) - 2)
        a, b = P[i], P[i + 1]
        t = b - a; L2 = (t ** 2).sum(1)
        u = np.clip(((X[:, :2] - a) * t).sum(1) / L2, -1.0, 2.0)
        foot = a + u[:, None] * t
        s = self.cs[i] + u * np.sqrt(L2)
        tn = t / np.sqrt(L2)[:, None]
        nvec = np.column_stack([-tn[:, 1], tn[:, 0]])          # left normal of the direction of increasing easting
        nvec = np.where(nvec[:, 1:2] < 0, -nvec, nvec)          # positive n points north on every segment
        n = ((X[:, :2] - foot) * nvec).sum(1)
        return np.column_stack([s, X[:, 2], n])
