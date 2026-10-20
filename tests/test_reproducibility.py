from dataclasses import replace
import json

import numpy as np
import pytest

from spp_unified import SingleParams, TwoSpeciesParams, SimulationConfig, simulate_single, simulate_two_species
from spp_unified.cli import run
from spp_unified.geometry import pairwise_displacements, resolve_overlaps, reflect_positions
from spp_unified.learned import behavior_descriptor, species_behavior_descriptor, _knn_overlap
from spp_unified.metrics import milling, heading_rotation
from spp_unified.neighbors import cell_list_pairs, dense_pairs
from spp_unified.parameters import ACTIVE_SINGLE_NAMES, TWO_SPECIES_EXTRA_NAMES, TWO_SPECIES_EXTRA_BOUNDS
from spp_unified.search import map_elites, parallel_map_elites


def toy_evaluator(x, seed):
    return float(x.sum()), tuple(x), {"evaluation_seed": seed}


def test_seed_replay_and_identical_species():
    p = SingleParams(rho=.08, noise=.2)
    cfg = SimulationConfig(steps=40, sample_start=20, seed=51)
    single = simulate_single(p, cfg, record_states=True)
    replay = simulate_single(p, cfg, record_states=True)
    two = simulate_two_species(TwoSpeciesParams(base=p), cfg, record_states=True)
    np.testing.assert_array_equal(single.states[-1][0], replay.states[-1][0])
    np.testing.assert_allclose(single.states[-1][0], two.states[-1][0], atol=1e-12)
    np.testing.assert_allclose(single.states[-1][1], two.states[-1][1], atol=1e-12)


@pytest.mark.parametrize("periodic", [True, False])
def test_sparse_neighbor_membership(periodic):
    points = np.random.default_rng(19).uniform(0, 4, (250, 2))
    dense = set(zip(*dense_pairs(points, 4, periodic, .6)))
    sparse = set(zip(*cell_list_pairs(points, 4, periodic, .6)))
    assert dense == sparse


def test_pair_projection_preserves_midpoint():
    x = np.array([[1., 1.], [1.1, 1.]])
    midpoint = x.mean(0)
    projected = resolve_overlaps(x.copy(), 4., .1, False)
    np.testing.assert_allclose(projected.mean(0), midpoint)
    assert np.linalg.norm(projected[0] - projected[1]) >= .2


def test_periodic_milling_translation():
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)]) * .4 + 2
    theta = angles + np.pi / 2
    assert milling(points, theta, 4., True) == pytest.approx(1.)
    assert milling((points + [2., .3]) % 4, theta, 4., True) == pytest.approx(1.)


def test_reflection_and_minimum_image():
    x, th = reflect_positions(np.array([[-.2, 1.], [2., 4.3]]), np.array([0., np.pi / 2]), 4.)
    np.testing.assert_allclose(x, [[.2, 1.], [2., 3.7]])
    np.testing.assert_allclose(np.column_stack([np.cos(th), np.sin(th)]), [[-1, 0], [0, -1]], atol=1e-12)
    _, distances = pairwise_displacements(np.array([[.1, 1], [3.9, 1]]), 4., True)
    assert distances[0, 1] == pytest.approx(.2)


def test_descriptor_shape_and_square_symmetries():
    cfg = SimulationConfig(steps=40, sample_start=20, seed=5)
    result = simulate_single(SingleParams(rho=.08), cfg, record_states=True, state_every=4)
    descriptor = behavior_descriptor(result.states, cfg)
    assert descriptor.shape == (51,)
    assert np.isfinite(descriptor).all()
    shifted = [((x + [.7, 1.1]) % cfg.S, th) for x, th in result.states]
    rotated = [(np.column_stack([-x[:, 1], x[:, 0]]) % cfg.S, th + np.pi / 2)
               for x, th in result.states]
    np.testing.assert_allclose(behavior_descriptor(shifted, cfg), descriptor, atol=2e-6)
    np.testing.assert_allclose(behavior_descriptor(rotated, cfg), descriptor, atol=2e-6)
    species = np.arange(len(result.states[0][0])) % 2
    assert species_behavior_descriptor(result.states, species, cfg).shape == (107,)


def test_rotation_estimator():
    slope = .04
    theta = np.tile(np.arange(60)[:, None] * slope, (1, 12))
    stats = heading_rotation(theta, np.ones(60))
    assert stats["omega"] == pytest.approx(slope)
    assert stats["rotation_r2"] == pytest.approx(1.)


def test_parameter_dimensions_and_validation():
    assert len(ACTIVE_SINGLE_NAMES) + len(TWO_SPECIES_EXTRA_NAMES) == 14
    assert TWO_SPECIES_EXTRA_BOUNDS.shape == (5, 2)
    with pytest.raises(ValueError, match="exactly 9"):
        SingleParams.from_active_vector(np.ones(8))
    with pytest.raises(ValueError, match="exactly 14"):
        TwoSpeciesParams.from_active_vector(np.ones(15))
    with pytest.raises(ValueError, match="boundary"):
        SimulationConfig(boundary="typo")
    with pytest.raises(ValueError, match="sample_start"):
        SimulationConfig(steps=10)


def test_species_swap_couplings():
    p = TwoSpeciesParams(alignment_symmetric=.4, alignment_antisymmetric=.7,
                         force_symmetric=-.3, force_antisymmetric=.2)
    q = replace(p, fraction_B=1-p.fraction_B,
                alignment_antisymmetric=-p.alignment_antisymmetric,
                force_antisymmetric=-p.force_antisymmetric)
    for a, b in zip(p.matrices(), q.matrices()):
        np.testing.assert_allclose(a[::-1, ::-1], b)


def test_knn_includes_closest_other_point():
    pytest.importorskip("sklearn")
    X = np.array([[0.], [.1], [1.], [3.], [4.], [6.]])
    Z = np.array([[0.], [2.9], [1.], [3.], [4.], [6.]])
    def nearest(a):
        distances = np.abs(a - a.T)
        np.fill_diagonal(distances, np.inf)
        return distances.argmin(axis=1)
    expected = np.mean(nearest(X) == nearest(Z))
    assert _knn_overlap(X, Z, k=1) == expected


def test_map_elites_exact_budget_and_replay():
    kw = dict(names=("x", "y"), bounds=np.array([[0., 1.], [0., 1.]]),
              measure_ranges=[(0., 1.), (0., 1.)], dims=(4, 4), evaluations=17,
              init_random=5, seed=37, record_every=5)
    a = map_elites(toy_evaluator, **kw)
    b = map_elites(toy_evaluator, **kw)
    assert a.elites.equals(b.elites)
    assert a.history.iloc[-1].evaluations == 17
    parallel = parallel_map_elites(toy_evaluator, batch_size=1, max_workers=1, **kw)
    assert parallel.elites.equals(a.elites)


def test_cli_writes_reloadable_outputs(tmp_path):
    record = run({"parameters": {"rho": .06},
                  "simulation": {"steps": 20, "sample_start": 10, "seed": 7}}, tmp_path)
    assert json.loads((tmp_path / "run.json").read_text())["summary"] == record["summary"]
    with np.load(tmp_path / "trajectory.npz", allow_pickle=False) as data:
        assert data["positions"].shape[0] == len(data["recorded_steps"])
        assert data["descriptor"].shape == (51,)
    with pytest.raises(ValueError, match="not empty"):
        run({}, tmp_path)


def test_autoencoder_train_statistics_and_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    from spp_unified.learned import BehaviorAutoencoder
    torch.set_num_threads(1)
    X = np.random.default_rng(3).normal(size=(50, 6)).astype(np.float32)
    X[35:] += 7
    model = BehaviorAutoencoder(6, hidden=(10, 5), seed=2)
    result = model.fit(X, np.arange(35), np.arange(35, 50), max_epochs=4)
    np.testing.assert_allclose(model.mean, X[:35].mean(0))
    assert np.isfinite(result.test_mse)
    path = tmp_path / "model.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=False)
    restored = BehaviorAutoencoder(6, hidden=(10, 5))
    restored.mean, restored.std = checkpoint["mean"], checkpoint["std"]
    restored.model.load_state_dict(checkpoint["state_dict"])
    np.testing.assert_array_equal(model.transform(X), restored.transform(X))


def test_compressed_boundary_cache(tmp_path):
    import gzip
    import pickle
    from boundary.cache import RunCache
    path = tmp_path / "cache.pkl"
    with gzip.open(str(path) + ".gz", "wb") as stream:
        pickle.dump({"a": np.array([1., 2.])}, stream)
    cache = RunCache(path)
    np.testing.assert_array_equal(cache.get("a"), [1., 2.])
    cache.put("b", 3)
    cache.flush()
    assert RunCache(path).get("b") == 3


def test_cma_mae_interface_and_accounting():
    pytest.importorskip("ribs")
    from spp_unified.search import cma_mae, archive_dataframe
    bounds = np.array([[0., 1.], [0., 1.]])
    result = cma_mae(toy_evaluator, 2, bounds, [(0, 1), (0, 1)],
                     iterations=2, emitters=2, batch_size=4, seed=7)
    assert result.history.iloc[-1].evaluations == 16
    table = archive_dataframe(result.archive, ("x", "y"), bounds)
    assert len(table) > 0
    assert table[["x", "y"]].min().min() >= 0
    assert table[["x", "y"]].max().max() <= 1
