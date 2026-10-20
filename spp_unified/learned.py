from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from .geometry import pairwise_displacements, relative_to_center


def _safe_quantiles(x: np.ndarray, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.zeros(len(qs), dtype=float)
    return np.quantile(x, qs).astype(float)


def _radial_shells(max_k: int = 4):
    groups: dict[int, list[tuple[int, int]]] = {}
    for kx in range(-max_k, max_k + 1):
        for ky in range(-max_k, max_k + 1):
            if kx == 0 and ky == 0:
                continue
            q = kx * kx + ky * ky
            groups.setdefault(q, []).append((kx, ky))
    return [groups[q] for q in sorted(groups)[:8]]


_K_SHELLS = _radial_shells(4)


def _structure_features(x: np.ndarray, S: float) -> np.ndarray:
    """Low-wave-number density spectrum, normalized to [0,1] scale."""
    n = max(len(x), 1)
    vals = []
    for shell in _K_SHELLS:
        shell_vals = []
        for kx, ky in shell:
            phase = 2.0 * np.pi * (kx * x[:, 0] + ky * x[:, 1]) / S
            amp = np.abs(np.exp(1j * phase).sum()) / n
            shell_vals.append(amp * amp)
        vals.append(float(np.mean(shell_vals)))
    return np.asarray(vals, dtype=float)


def _periodic_displacement(a: np.ndarray, b: np.ndarray, S: float) -> np.ndarray:
    d = b - a
    return d - S * np.round(d / S)


def behavior_descriptor(states: list[tuple[np.ndarray, np.ndarray]], cfg, *, nbins: int = 12,
                        max_frames: int = 10) -> np.ndarray:
    """Translation- and rotation-invariant descriptor learned from raw trajectories.

    The descriptor deliberately does not receive the hand-designed order parameters.
    It combines pair-distance statistics, radial heading correlations, low-k density
    modes, local-density statistics, shape, and short-time dynamics.
    """
    if not states:
        raise ValueError("states must contain at least one frame")
    frames = states[-max_frames:]
    periodic = cfg.boundary == "periodic"
    rmax = cfg.S / 2.0 if periodic else cfg.S / np.sqrt(2.0)
    edges = np.linspace(0.0, rmax, nbins + 1)
    pair_hist = np.zeros(nbins)
    pair_corr = np.zeros(nbins)
    pair_count = np.zeros(nbins)
    nn_all, local_cv, shape_aniso, rgyr = [], [], [], []
    sk = []

    for x, th in frames:
        n = len(x)
        _, r = pairwise_displacements(x, cfg.S, periodic)
        iu = np.triu_indices(n, 1)
        rr = r[iu]
        cc = np.cos(th[:, None] - th[None, :])[iu]
        idx = np.digitize(rr, edges) - 1
        for b in range(nbins):
            sel = idx == b
            if np.any(sel):
                pair_hist[b] += float(sel.sum())
                pair_corr[b] += float(cc[sel].sum())
                pair_count[b] += float(sel.sum())

        rf = r.copy()
        np.fill_diagonal(rf, np.inf)
        nn = np.min(rf, axis=1)
        nn_all.extend(nn[np.isfinite(nn)].tolist())
        rc = 4.0 * cfg.r0
        counts = (rf < rc).sum(axis=1).astype(float)
        local_cv.append(float(np.std(counts) / (np.mean(counts) + 1e-8)))

        _, rel = relative_to_center(x, cfg.S, periodic)
        C = (rel.T @ rel) / max(n, 1)
        ev = np.linalg.eigvalsh(C)
        shape_aniso.append(float((ev[-1] - ev[0]) / (ev.sum() + 1e-9)))
        rgyr.append(float(np.sqrt(np.trace(C)) / cfg.S))
        sk.append(_structure_features(x, cfg.S))

    # Divide pair counts by the ideal-gas shell expectation. This is a proper
    # finite-system radial distribution estimate for periodic boxes; for walls it
    # is an approximate normalization and is used only for representation learning.
    n = len(frames[-1][0])
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    expected_pairs_per_frame = n * (n - 1) / 2.0 * shell_area / (cfg.S * cfg.S)
    g = pair_hist / (len(frames) * expected_pairs_per_frame + 1e-9)
    g = np.log1p(np.clip(g, 0.0, 20.0)) / np.log(21.0)
    corr = np.divide(pair_corr, pair_count, out=np.zeros_like(pair_corr), where=pair_count > 0)

    # Short-time dynamics from recorded frames.
    pol = np.asarray([np.mean(np.exp(1j * th)) for _, th in frames])
    pol_mag = np.abs(pol)
    pol_ang = np.unwrap(np.angle(pol))
    dyn = []
    for lag in (1, 2, 4):
        if len(pol) > lag:
            z = pol[lag:] * np.conj(pol[:-lag])
            dyn.extend([float(np.mean(np.abs(z))), float(np.abs(np.mean(z)))])
        else:
            dyn.extend([0.0, 0.0])
    if len(pol_ang) > 1:
        da = np.diff(pol_ang)
        dyn.extend([float(np.mean(np.abs(da)) / np.pi), float(np.std(da) / np.pi)])
    else:
        dyn.extend([0.0, 0.0])

    # Mobility is computed from minimum-image frame-to-frame displacements.
    mobility = []
    if len(frames) > 1:
        for (x0, _), (x1, _) in zip(frames[:-1], frames[1:]):
            d = _periodic_displacement(x0, x1, cfg.S) if periodic else (x1 - x0)
            mobility.append(float(np.mean(np.linalg.norm(d, axis=1)) / (cfg.S + 1e-9)))
    mob = [float(np.mean(mobility)) if mobility else 0.0,
           float(np.std(mobility)) if mobility else 0.0]

    extras = np.concatenate([
        _safe_quantiles(np.asarray(nn_all) / cfg.S),
        np.asarray([np.mean(local_cv), np.std(local_cv), np.mean(shape_aniso),
                    np.mean(rgyr), np.mean(pol_mag), np.std(pol_mag)], dtype=float),
        np.mean(np.asarray(sk), axis=0),
        np.asarray(dyn + mob, dtype=float),
    ])
    return np.concatenate([g, corr, extras]).astype(np.float32)


def species_behavior_descriptor(states: list[tuple[np.ndarray, np.ndarray]], species: np.ndarray,
                                cfg, *, nbins: int = 8, max_frames: int = 10) -> np.ndarray:
    """Species-aware extension of :func:`behavior_descriptor` for two species."""
    base = behavior_descriptor(states, cfg, nbins=12, max_frames=max_frames)
    frames = states[-max_frames:]
    periodic = cfg.boundary == "periodic"
    rmax = cfg.S / 2.0 if periodic else cfg.S / np.sqrt(2.0)
    edges = np.linspace(0.0, rmax, nbins + 1)
    pair_types = [(0, 0), (1, 1), (0, 1)]
    hist = np.zeros((3, nbins))
    corr = np.zeros((3, nbins))
    cnt = np.zeros((3, nbins))
    rel_pol = []
    rel_com = []

    for x, th in frames:
        _, r = pairwise_displacements(x, cfg.S, periodic)
        cmat = np.cos(th[:, None] - th[None, :])
        for pidx, (a, b) in enumerate(pair_types):
            if a == b:
                ii, jj = np.where(np.triu((species[:, None] == a) & (species[None, :] == b), 1))
            else:
                ii, jj = np.where((species[:, None] == a) & (species[None, :] == b))
            rr = r[ii, jj]
            cc = cmat[ii, jj]
            idx = np.digitize(rr, edges) - 1
            for q in range(nbins):
                sel = idx == q
                if np.any(sel):
                    hist[pidx, q] += float(sel.sum())
                    corr[pidx, q] += float(cc[sel].sum())
                    cnt[pidx, q] += float(sel.sum())

        z = []
        centers = []
        for a in (0, 1):
            sel = species == a
            z.append(np.mean(np.exp(1j * th[sel])) if np.any(sel) else 0j)
            centers.append(np.mean(x[sel], axis=0) if np.any(sel) else np.zeros(2))
        rel_pol.extend([abs(z[0]), abs(z[1]), np.cos(np.angle(z[0]) - np.angle(z[1]))])
        dc = _periodic_displacement(centers[0], centers[1], cfg.S) if periodic else centers[1] - centers[0]
        rel_com.append(float(np.linalg.norm(dc) / cfg.S))

    hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-9)
    corr = np.divide(corr, cnt, out=np.zeros_like(corr), where=cnt > 0)
    rel_pol = np.asarray(rel_pol, dtype=float).reshape(-1, 3)
    extra = np.concatenate([hist.ravel(), corr.ravel(), rel_pol.mean(axis=0),
                            rel_pol.std(axis=0), np.asarray([np.mean(rel_com), np.std(rel_com)])])
    return np.concatenate([base, extra.astype(np.float32)]).astype(np.float32)


def hexatic_order(x: np.ndarray, S: float, periodic: bool, k: int = 6) -> float:
    n = len(x)
    if n <= 2:
        return 0.0
    d, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    kk = min(k, n - 1)
    idx = np.argpartition(r, kk - 1, axis=1)[:, :kk]
    rows = np.arange(n)[:, None]
    vec = d[rows, idx]
    ang = np.arctan2(vec[:, :, 1], vec[:, :, 0])
    psi = np.mean(np.exp(6j * ang), axis=1)
    return float(np.abs(np.mean(psi)))


def density_mode_peak(x: np.ndarray, S: float) -> float:
    return float(np.max(_structure_features(x, S)))


def local_density_cv(x: np.ndarray, radius: float, S: float, periodic: bool) -> float:
    _, r = pairwise_displacements(x, S, periodic)
    np.fill_diagonal(r, np.inf)
    c = (r < radius).sum(axis=1).astype(float)
    return float(np.std(c) / (np.mean(c) + 1e-9))


@dataclass
class AutoencoderResult:
    train_mse: float
    test_mse: float
    trustworthiness: float
    knn_overlap: float
    epochs: int


class BehaviorAutoencoder:
    """Small deterministic PyTorch autoencoder with a two-dimensional bottleneck."""
    def __init__(self, input_dim: int, latent_dim: int = 2, hidden=(64, 24), seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.seed = seed
        h1, h2 = hidden
        self.encoder = nn.Sequential(nn.Linear(input_dim, h1), nn.Tanh(),
                                     nn.Linear(h1, h2), nn.Tanh(),
                                     nn.Linear(h2, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, h2), nn.Tanh(),
                                     nn.Linear(h2, h1), nn.Tanh(),
                                     nn.Linear(h1, input_dim))
        self.model = nn.Sequential(self.encoder, self.decoder)
        self.mean = None
        self.std = None
        self.result: AutoencoderResult | None = None

    def fit(self, X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
            *, max_epochs: int = 500, patience: int = 50, lr: float = 2e-3,
            weight_decay: float = 1e-5) -> AutoencoderResult:
        import torch
        from sklearn.manifold import trustworthiness
        X = np.asarray(X, dtype=np.float32)
        self.mean = X[train_idx].mean(axis=0)
        self.std = X[train_idx].std(axis=0)
        self.std = np.maximum(self.std, 1e-6)
        Xs = (X - self.mean) / self.std
        xt = torch.tensor(Xs, dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()
        best = np.inf
        best_state = None
        wait = 0
        epochs = 0
        for ep in range(max_epochs):
            self.model.train()
            opt.zero_grad()
            pred = self.model(xt[train_idx])
            loss = loss_fn(pred, xt[train_idx])
            loss.backward()
            opt.step()
            self.model.eval()
            with torch.no_grad():
                val = float(loss_fn(self.model(xt[test_idx]), xt[test_idx]).item())
            epochs = ep + 1
            if val < best - 1e-5:
                best = val
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            rec = self.model(xt).numpy()
            Z = self.encoder(xt).numpy()
        train_mse = float(np.mean((rec[train_idx] - Xs[train_idx]) ** 2))
        test_mse = float(np.mean((rec[test_idx] - Xs[test_idx]) ** 2))
        tw = float(trustworthiness(Xs[test_idx], Z[test_idx], n_neighbors=min(10, max(2, len(test_idx)//4))))
        overlap = _knn_overlap(Xs[test_idx], Z[test_idx], k=min(10, max(2, len(test_idx)//4)))
        self.result = AutoencoderResult(train_mse, test_mse, tw, overlap, epochs)
        return self.result

    def transform(self, X: np.ndarray) -> np.ndarray:
        import torch
        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self.mean) / self.std
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(torch.tensor(Xs)).numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        import torch
        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self.mean) / self.std
        self.model.eval()
        with torch.no_grad():
            rec = self.model(torch.tensor(Xs)).numpy()
        return rec * self.std + self.mean

    def save(self, path: str):
        import torch
        torch.save({"seed": self.seed, "mean": self.mean, "std": self.std,
                    "state_dict": self.model.state_dict()}, path)


def _knn_overlap(X: np.ndarray, Z: np.ndarray, k: int = 10) -> float:
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(X) - 1)
    if k <= 0:
        return 1.0
    # With X=None sklearn already excludes each observation itself.
    nx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    nz = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(return_distance=False)
    vals = [len(set(a).intersection(set(b))) / k for a, b in zip(nx, nz)]
    return float(np.mean(vals))


def normalize_latent(Z: np.ndarray, lo=None, hi=None):
    Z = np.asarray(Z, dtype=float)
    if lo is None:
        lo = np.percentile(Z, 1, axis=0)
    if hi is None:
        hi = np.percentile(Z, 99, axis=0)
    hi = np.where(np.asarray(hi) - np.asarray(lo) < 1e-9, np.asarray(lo) + 1.0, hi)
    return np.clip((Z - lo) / (hi - lo), 0.0, 1.0), np.asarray(lo), np.asarray(hi)
