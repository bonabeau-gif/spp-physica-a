"""Module B: protocol dataclasses, tau_int estimation, adaptive run-length policy."""
from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import numpy as np


@dataclass(frozen=True)
class Protocol:
    space: str = "two"               # "single" | "two"
    boundary: str = "periodic"       # "periodic" | "reflecting"
    n_steps: int = 220               # may be raised by run_adequate
    burn_in_frac: float = 0.5
    max_steps: int = 3520            # hard cap for the adaptive policy

    def hash(self) -> str:
        s = f"{self.space}|{self.boundary}|{self.n_steps}|{self.burn_in_frac}"
        return hashlib.blake2b(s.encode(), digest_size=8).hexdigest()

    def with_steps(self, n_steps: int) -> "Protocol":
        return replace(self, n_steps=int(n_steps))


def u_hash(u: np.ndarray) -> str:
    q = np.round(np.asarray(u, float), 9)
    return hashlib.sha1(q.tobytes()).hexdigest()


def derive_seed(uh: str, ph: str, seed_index: int) -> int:
    """Independent, reproducible seed stream per (u, protocol, seed_index)."""
    h = hashlib.blake2b(f"{uh}:{ph}:{seed_index}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


def tau_int(series: np.ndarray, c: float = 5.0) -> float:
    """Sokal automatic-windowing integrated autocorrelation time (in sample units)."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4 or np.std(x) < 1e-12:
        return 0.5
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var <= 0:
        return 0.5
    # normalized autocovariance via FFT
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, m)
    acf = np.fft.irfft(f * np.conj(f), m)[:n].real / (var * n)
    tau = 0.5
    for w in range(1, n):
        tau += acf[w]
        if w >= c * tau:
            break
    return float(max(tau, 0.5))


# ---- adequacy on RunResult.tau_int (in step units) ----
def adequate(steps_used: int, burn_in_frac: float, tau_by_key: dict, keys: list,
             min_tau_multiples: float = 50.0) -> bool:
    taus = [tau_by_key[k] for k in keys if k in tau_by_key and np.isfinite(tau_by_key[k])]
    if not taus:
        return False
    tmax = max(taus)
    window_length = steps_used * (1.0 - burn_in_frac)
    burn_in = steps_used * burn_in_frac
    return (window_length >= min_tau_multiples * tmax) and (burn_in >= 10.0 * tmax)
