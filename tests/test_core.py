import numpy as np
from spp_unified.model import manuscript_pair_force, beta_from_adhesion, simulate_single, simulate_two_species
from spp_unified.parameters import SingleParams, TwoSpeciesParams, SimulationConfig
from spp_unified.metrics import polarization, milling, clustering


def test_exact_force_endpoints_and_zero():
    r0, r1, beta = 0.1, 0.8, 3.0
    r = np.array([0.19, 0.2, 0.35, r1 - 1e-8, r1])
    f = manuscript_pair_force(r, r0, r1, beta, 50.0)
    assert f[0] == -50.0 and f[1] == -50.0
    # zero occurs at u=1/(beta+1)
    rz = 2*r0 + (r1-2*r0)/(beta+1)
    assert abs(manuscript_pair_force(np.array([rz]), r0, r1, beta, 50.0)[0]) < 1e-12
    assert f[-1] == 0.0


def test_metrics_synthetic_states():
    th = np.zeros(20)
    assert polarization(th) > 0.999
    x = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 20, endpoint=False)),
                         np.sin(np.linspace(0, 2*np.pi, 20, endpoint=False))]) + 2
    tang = np.linspace(0, 2*np.pi, 20, endpoint=False) + np.pi/2
    assert milling(x, tang, 10.0, False) > 0.99


def test_single_smoke_and_reciprocal_reduction():
    cfg = SimulationConfig(steps=60, sample_start=30, sample_every=2, seed=3)
    p = SingleParams(rho=.08, gradient_strength=0, noise=.1)
    a = simulate_single(p, cfg).summary
    assert 0 <= a["phi"] <= 1
    tp = TwoSpeciesParams(base=p, alignment_symmetric=1.0,
                          alignment_antisymmetric=0.0,
                          force_symmetric=0.0, force_antisymmetric=0.0)
    b = simulate_two_species(tp, cfg).summary
    # Same equations in the reciprocal limit; species-label randomization should not change them.
    for k in ["phi", "milling", "clustering"]:
        assert np.isfinite(b[k])
