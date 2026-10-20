"""Monkeypatch BehaviorAutoencoder.transform with a pure-NumPy forward pass.

torch 2.2.2 (the last Intel-macOS wheel) is incompatible with NumPy 2.x, so its
tensor<->numpy bridge is dead and ae.transform() raises. The encoder is a tiny
fixed tanh MLP (51->64->24->2); we extract its weights via .tolist() (which does
not use the broken bridge) and evaluate the forward pass in NumPy. Numerically
identical to the torch forward up to float32 rounding. Import this module before
calling ae.transform().
"""
from __future__ import annotations
import numpy as np
from spp_unified import learned

_KEYS = ["0.0.weight", "0.0.bias", "0.2.weight", "0.2.bias", "0.4.weight", "0.4.bias"]


def _np_transform(self, X):
    if not hasattr(self, "_npw"):
        sd = self.model.state_dict()
        self._npw = [np.asarray(sd[k].tolist(), dtype=np.float32) for k in _KEYS]
    W1, b1, W2, b2, W3, b3 = self._npw
    X = np.asarray(X, dtype=np.float32)
    Xs = (X - self.mean) / self.std
    h = np.tanh(Xs @ W1.T + b1)
    h = np.tanh(h @ W2.T + b2)
    return (h @ W3.T + b3).astype(np.float32)


learned.BehaviorAutoencoder.transform = _np_transform
