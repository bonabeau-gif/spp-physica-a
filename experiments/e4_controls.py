"""E4: control battery replacement.

For each cataloged boundary type, apply three controls instead of the vacuous f_B->0:
  C1 matched reciprocal: hold J_a=C_a=0 (f_B held) -> does the dominant-observable jump survive?
  C2 single-species subspace: take the 9 single-species parameters and test whether a
     descriptor boundary exists there under the single-species oracle.
  C3 species-swap: exchange A<->B (J_a->-J_a, C_a->-C_a, f_B->1-f_B, and the normal's
     J_a/C_a/f_B components negated). For a symmetric model the boundary must be invariant
     up to seed noise; a C3 failure is a species-asymmetric implementation bug -> BLOCKING STOP.
f_B->0 retained only as a labeled sanity check.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pickle
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary import config as CFG
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.params import clip_unit, idx
from boundary.sensitivity import sensitivity
from boundary.catalog import dominant_observable, _obs_jump, _phys_to_unit

OUT = ROOT / "results" / "boundary"
JUMP_SURVIVE = 0.4          # dominant-obs jump ratio threshold (same as catalog controls)
SWAP_TOL = 0.5              # C3: |jump_swap/jump_orig - 1| must be within this (symmetry)


def load_types(sp):
    types = []
    c2 = pickle.load(open(OUT / "campaign2.pkl", "rb"))
    ja = [i for i, f in enumerate(c2["family"]) if f == "A_chiral_Ja"]
    jb = [c2["susc"][f"A_chiral_Ja_{k}"]["b"] for k in range(len(ja))]
    r = ja[int(np.argmax(jb))]
    types.append(("chiral_Ja", np.array(c2["points"][r]), np.array(c2["normals"][r])))
    tr = pickle.load(open(OUT / "traces.pkl", "rb"))
    bi = int(np.argmax([b for t in tr["traces"] for b in t["bvals"]]))
    types.append(("adhesion", np.array(tr["points"][bi]), np.array(tr["normals"][bi])))
    c3 = pickle.load(open(OUT / "campaign3.pkl", "rb"))
    for fam in ("D_demix_Cs", "E_demix_Ca"):
        m = [i for i, f in enumerate(c3["family"]) if f == fam]
        if m:
            rr = max(m, key=lambda i: c3["susc"][i]["b"])
            types.append((fam, np.array(c3["points"][rr]), np.array(c3["normals"][rr])))
    return types


def c2_single_subspace(sp_two, u_two, proto_single, cache_s):
    """Does a descriptor boundary exist at the same 9 single-species parameters?"""
    sp1 = P.get_space("single")
    phys_two = P.from_unit(u_two, sp_two)
    phys9 = np.array([phys_two[sp_two.param_order.index(n)] for n in sp1.param_order])
    u1 = clip_unit(P.to_unit(phys9, sp1))
    s = sensitivity(sp1, u1, proto_single, h=0.04, n_side=CFG.N_SIDE, cache=cache_s)
    return {"b_single": float(s.b), "found": bool(s.b >= CFG.B_MIN)}


def c3_species_swap(sp, u, normal, proto, cache, dom_obs, orig_jump):
    phys = P.from_unit(u, sp)
    ph = dict(zip(sp.param_order, phys))
    ph["J_a"] = -ph["J_a"]; ph["C_a"] = -ph["C_a"]; ph["f_B"] = 1.0 - ph["f_B"]
    u_sw = clip_unit(P.to_unit(np.array([ph[n] for n in sp.param_order]), sp))
    n_sw = np.asarray(normal, float).copy()
    for nm in ("J_a", "C_a", "f_B"):
        n_sw[idx(sp, nm)] *= -1.0
    jump_sw = _obs_jump(sp, u_sw, n_sw, proto, cache, dom_obs)
    ratio = jump_sw / orig_jump if orig_jump > 1e-6 else np.nan
    return {"jump_swap": float(jump_sw), "ratio": float(ratio),
            "invariant": bool(np.isfinite(ratio) and abs(ratio - 1.0) <= SWAP_TOL)}


def classify(c1_survives, c2_found):
    if (not c1_survives) and (not c2_found):
        return "genuine non-reciprocity boundary"
    if c1_survives and c2_found:
        return "single-species boundary in disguise"
    if c1_survives and (not c2_found):
        return "reciprocal mixture boundary (composition/cross-coupling)"
    return "investigate (coincidence of two boundaries)"


def main():
    sp = P.get_space("two")
    proto = Protocol(space="two", n_steps=240)
    proto_s = Protocol(space="single", n_steps=CFG.BASE_STEPS["single"])
    cache = RunCache(OUT / "cache_two.pkl")
    cache_s = RunCache(OUT / "cache_single.pkl")
    out = {"config_hash": CFG.CONFIG_HASH, "types": [], "c3_all_invariant": True}
    for name, u, n in load_types(sp):
        u = clip_unit(u)
        dom = dominant_observable(sp, u, n, proto, cache=cache, n_side=16)
        do = dom["dominant_obs"]
        orig_jump = _obs_jump(sp, u, n, proto, cache, do)
        # C1 matched reciprocal
        uc = u.copy(); uc[idx(sp, "J_a")] = _phys_to_unit(sp, "J_a", 0.0); uc[idx(sp, "C_a")] = _phys_to_unit(sp, "C_a", 0.0)
        jc1 = _obs_jump(sp, clip_unit(uc), n, proto, cache, do)
        c1_survives = (jc1 / orig_jump >= JUMP_SURVIVE) if orig_jump > 1e-6 else True
        # C2 single-species subspace
        c2 = c2_single_subspace(sp, u, proto_s, cache_s)
        # C3 species-swap (blocking gate)
        c3 = c3_species_swap(sp, u, n, proto, cache, do, orig_jump)
        out["c3_all_invariant"] = out["c3_all_invariant"] and c3["invariant"]
        # f_B->0 sanity check (labeled)
        uf = u.copy(); uf[idx(sp, "f_B")] = _phys_to_unit(sp, "f_B", 0.0)
        jf = _obs_jump(sp, clip_unit(uf), n, proto, cache, do)
        label = classify(c1_survives, c2["found"])
        rec = {"type": name, "dominant_obs": do, "orig_jump": float(orig_jump),
               "C1_reciprocal": {"jump_ratio": float(jc1 / orig_jump) if orig_jump > 1e-6 else None, "survives": bool(c1_survives)},
               "C2_single_subspace": c2,
               "C3_species_swap": c3,
               "fB0_sanity": {"jump_ratio": float(jf / orig_jump) if orig_jump > 1e-6 else None},
               "label": label}
        out["types"].append(rec)
        cache.flush(); cache_s.flush()
        print(f"{name}: dom={do} | C1 recip survives={c1_survives}({jc1/max(orig_jump,1e-9):.2f}) | "
              f"C2 single found={c2['found']}(b={c2['b_single']:.2f}) | "
              f"C3 swap invariant={c3['invariant']}(ratio={c3['ratio']:.2f}) | => {label}", flush=True)
    json.dump(out, open(OUT / "E4_controls.json", "w"), indent=2)
    print(f"\nE4 c3_all_invariant={out['c3_all_invariant']}", flush=True)
    if not out["c3_all_invariant"]:
        print("STOP: C3 species-swap failed on a type -> species-asymmetric implementation. BLOCKING.", flush=True)


if __name__ == "__main__":
    main()
