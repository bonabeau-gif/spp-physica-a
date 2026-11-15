"""Parameter, model, and output registry for the command-line layer.

This module is the single source of truth for everything a run configuration can
set and everything a run records. ``describe``, ``template``, ``validate``, the
committed JSON Schema (``docs/config.schema.json``), and the run-directory
``schema.json`` are all generated from it.

Defaults mirror the dataclasses in :mod:`spp_unified.parameters` and the CLI
default ``state_every=8``; ``tests/test_agent_layer.py`` checks that they agree.
Validity ranges come from the ``__post_init__`` checks in ``parameters.py``.
``explored_range`` records the finite search bounds used in the paper
(``EXPERIMENT_BOUNDS``, ``TWO_SPECIES_EXTRA_BOUNDS``); values outside them are
valid but were not studied. Where the source states no range, ``range`` is None.

This module imports nothing heavy so that ``describe`` and ``--help`` stay fast.
"""
from __future__ import annotations

REGISTRY_VERSION = 1
CONFIG_SCHEMA_VERSION = 1

DIMENSIONLESS = "dimensionless"
LENGTH = "simulation length unit"
TIME = "simulation time unit (one update has duration dt)"

MODELS = {
    "single": {
        "description": (
            "Single-species self-propelled particles in a square box. Each synchronous "
            "update combines heading persistence (inertia), normalized local alignment "
            "including self, an optional constant taxis cue, and a piecewise radial pair "
            "force, then adds uniform angular noise and moves particles at speed v0. "
            "Three overlap-projection passes follow each update."),
        "groups": ["parameters", "simulation", "state_every"],
        "library_call": "spp_unified.simulate_single(SingleParams, SimulationConfig)",
    },
    "two": {
        "description": (
            "Two-species extension. Species B is a fraction fraction_B of the particles. "
            "Cross-species alignment weights J=[[1, Js+Ja],[Js-Ja, 1]] and additive radial "
            "couplings C=[[0, Cs+Ca],[Cs-Ca, 0]] are set by symmetric (s) and "
            "antisymmetric (a) parts. Ja=Ca=0 is the matched reciprocal control; with "
            "alignment_symmetric=1 and zero force couplings it reduces to the single model."),
        "groups": ["parameters", "mixture", "simulation", "state_every"],
        "library_call": "spp_unified.simulate_two_species(TwoSpeciesParams, SimulationConfig)",
    },
}
BOTH = ["single", "two"]

# ---------------------------------------------------------------------------
# Parameters. ``name`` is the dotted key in a configuration file.
# ---------------------------------------------------------------------------
PARAMETERS = [
    {"name": "kind", "type": "enum", "default": "single", "options": ["single", "two"],
     "unit": None, "applies_to": BOTH, "constraints": [],
     "description": "Model to run: 'single' species or 'two' species."},

    # --- particle rules: SingleParams ---
    {"name": "parameters.rho", "type": "number", "default": 0.15,
     "range": {"min": 0.0, "min_exclusive": True, "max": None},
     "explored_range": [0.05, 0.30], "unit": DIMENSIONLESS, "applies_to": BOTH, "constraints": [],
     "paper_symbol": "rho",
     "description": ("Packing density relative to hexagonal close packing. The particle count is "
                     "n = max(8, round(rho * round(sqrt(3)/6 * (S/r0)^2))), so rho sets n, not a "
                     "number per unit area. Higher rho gives more particles and quadratically "
                     "longer runs. Very high rho can make initial non-overlapping placement fail. "
                     "The single-species learned corpus used rho up to 0.70 at S=3.2.")},
    {"name": "parameters.v0", "type": "number", "default": 0.05,
     "range": {"min": 0.0, "max": None}, "explored_range": [0.01, 0.12],
     "unit": "simulation length unit per simulation time unit", "applies_to": BOTH,
     "constraints": [], "paper_symbol": "v0",
     "description": "Self-propulsion speed. Each update moves a particle v0*dt along its new heading."},
    {"name": "parameters.inertia", "type": "number", "default": 0.5, "range": None,
     "explored_range": [0.0, 5.0], "unit": DIMENSIONLESS, "applies_to": BOTH, "constraints": [],
     "paper_symbol": "epsilon",
     "description": "Weight of the current heading in the drive (heading persistence)."},
    {"name": "parameters.interaction_range_ratio", "type": "number", "default": 7.0,
     "range": {"min": 2.0, "min_exclusive": True, "max": None}, "explored_range": [2.1, 18.0],
     "unit": "multiples of r0", "applies_to": BOTH, "paper_symbol": "r1/r0",
     "constraints": ["r1_within_half_box"],
     "description": ("Pair-force cutoff r1 in units of r0. The force is repulsive just outside the "
                     "hard core (2*r0), crosses zero, and is attractive up to r1. Also the radius "
                     "used for the clustering and segregation measures.")},
    {"name": "parameters.force_strength", "type": "number", "default": 1.0, "range": None,
     "explored_range": [0.0, 5.0], "unit": DIMENSIONLESS, "applies_to": BOTH, "constraints": [],
     "paper_symbol": "f0",
     "description": "Weight of the summed radial pair force in the drive. 0 disables pair forces in the heading update (overlap projection still applies)."},
    {"name": "parameters.adhesion", "type": "number", "default": 0.0, "range": None,
     "explored_range": [-0.80, 0.80], "unit": DIMENSIONLESS, "applies_to": BOTH,
     "constraints": ["adhesion_clip"], "paper_symbol": "B",
     "description": ("Attraction parameter B; beta = exp(tan(pi*B/2)) sets the attractive part of "
                     "the force law. Larger B means stronger attraction. The code clips B to "
                     "[-0.95, 0.95] (and tan to [-8, 8]) before mapping, so values beyond that "
                     "have no further effect.")},
    {"name": "parameters.alignment_range_ratio", "type": "number", "default": 7.0,
     "range": {"min": 0.0, "min_exclusive": True, "max": None}, "explored_range": [2.1, 18.0],
     "unit": "multiples of r0", "applies_to": BOTH, "constraints": [], "paper_symbol": "ra/r0",
     "description": "Alignment radius ra in units of r0: neighbors within ra contribute to the normalized alignment direction."},
    {"name": "parameters.alignment_strength", "type": "number", "default": 2.0, "range": None,
     "explored_range": [0.0, 5.0], "unit": DIMENSIONLESS, "applies_to": BOTH, "constraints": [],
     "paper_symbol": "alpha",
     "description": "Weight of the local alignment direction in the drive. Larger values favor polar order (flocking)."},
    {"name": "parameters.gradient_angle", "type": "number", "default": 0.0, "range": None,
     "explored_range": None, "unit": "radian", "applies_to": BOTH, "constraints": [],
     "paper_symbol": "angle of G",
     "description": "Direction of the constant taxis cue G. Fixed in the paper's searches (taxis strength 0)."},
    {"name": "parameters.gradient_strength", "type": "number", "default": 0.0, "range": None,
     "explored_range": None, "unit": DIMENSIONLESS, "applies_to": BOTH, "constraints": [],
     "paper_symbol": "gamma",
     "description": "Weight of the taxis cue in the drive. The paper uses gamma = 0."},
    {"name": "parameters.noise", "type": "number", "default": 0.10,
     "range": {"min": 0.0, "max": 1.0}, "explored_range": [0.0, 1.0],
     "unit": "fraction of pi (angular noise amplitude)", "applies_to": BOTH, "constraints": [],
     "paper_symbol": "eta",
     "description": ("Angular noise amplitude: each update adds noise * U[-pi, pi] to the heading. "
                     "Increasing noise destroys polar order (order-disorder transition).")},

    # --- mixture: TwoSpeciesParams ---
    {"name": "mixture.fraction_B", "type": "number", "default": 0.5,
     "range": {"min": 0.0, "max": 1.0}, "explored_range": [0.20, 0.80], "unit": DIMENSIONLESS,
     "applies_to": ["two"], "constraints": [], "paper_symbol": "f_B",
     "description": "Fraction of particles assigned to species B (n_B = round(fraction_B * n)). Labels are shuffled with seed+7919."},
    {"name": "mixture.alignment_symmetric", "type": "number", "default": 1.0, "range": None,
     "explored_range": [-1.0, 1.5], "unit": DIMENSIONLESS, "applies_to": ["two"], "constraints": [],
     "paper_symbol": "J_s",
     "description": "Symmetric part of the cross-species alignment weight. 1 with J_a=0 makes species indistinguishable in alignment."},
    {"name": "mixture.alignment_antisymmetric", "type": "number", "default": 0.0, "range": None,
     "explored_range": [-1.5, 1.5], "unit": DIMENSIONLESS, "applies_to": ["two"], "constraints": [],
     "paper_symbol": "J_a",
     "description": "Antisymmetric (non-reciprocal) part of the cross-species alignment weight: A aligns to B with Js+Ja, B to A with Js-Ja."},
    {"name": "mixture.force_symmetric", "type": "number", "default": 0.0, "range": None,
     "explored_range": [-1.0, 1.0], "unit": DIMENSIONLESS, "applies_to": ["two"], "constraints": [],
     "paper_symbol": "C_s",
     "description": "Symmetric part of the additive cross-species radial coupling (linear envelope from 2*r0 to r1). Positive attracts."},
    {"name": "mixture.force_antisymmetric", "type": "number", "default": 0.0, "range": None,
     "explored_range": [-1.5, 1.5], "unit": DIMENSIONLESS, "applies_to": ["two"], "constraints": [],
     "paper_symbol": "C_a",
     "description": "Antisymmetric (non-reciprocal) part of the additive cross-species radial coupling."},

    # --- simulation: SimulationConfig ---
    {"name": "simulation.S", "type": "number", "default": 4.0,
     "range": None, "explored_range": None, "unit": LENGTH,
     "applies_to": BOTH, "constraints": ["box_exceeds_diameter", "r1_within_half_box"],
     "description": "Side length of the square box. With r0 fixed, larger S means more particles at the same rho (cost grows as n^2). The paper uses S=4 (single-species learned corpus: S=3.2)."},
    {"name": "simulation.r0", "type": "number", "default": 0.1,
     "range": {"min": 0.0, "min_exclusive": True, "max": None}, "explored_range": None,
     "unit": LENGTH, "applies_to": BOTH, "constraints": ["box_exceeds_diameter"],
     "description": "Particle radius; hard-core diameter is 2*r0. Interaction radii are multiples of r0."},
    {"name": "simulation.steps", "type": "integer", "default": 240,
     "range": {"min": 1, "max": None}, "explored_range": None, "unit": "updates",
     "applies_to": BOTH, "constraints": ["sample_start_before_steps"],
     "description": "Number of synchronous updates. Runtime is linear in steps."},
    {"name": "simulation.sample_start", "type": "integer", "default": 120,
     "range": {"min": 0, "max": None}, "explored_range": None, "unit": "updates",
     "applies_to": BOTH, "constraints": ["sample_start_before_steps"],
     "description": "Zero-based loop index at which order-parameter sampling begins (burn-in). Summary values average only over samples at or after it."},
    {"name": "simulation.sample_every", "type": "integer", "default": 2,
     "range": {"min": 1, "max": None}, "explored_range": None, "unit": "updates",
     "applies_to": BOTH, "constraints": [],
     "description": "Sampling interval for the order-parameter time series."},
    {"name": "simulation.boundary", "type": "enum", "default": "periodic",
     "options": ["periodic", "reflecting"], "unit": None, "applies_to": BOTH, "constraints": [],
     "description": "Box boundary: periodic (minimum image) or reflecting walls (normal heading component reversed)."},
    {"name": "simulation.dt", "type": "number", "default": 1.0,
     "range": {"min": 0.0, "min_exclusive": True, "max": None}, "explored_range": None,
     "unit": TIME, "applies_to": BOTH, "constraints": [],
     "description": "Time step. The paper uses dt=1. Changes displacement per update (v0*dt), not the number of updates."},
    {"name": "simulation.hard_core_strength", "type": "number", "default": 50.0,
     "range": {"min": 0.0, "min_exclusive": True, "max": None}, "explored_range": None,
     "unit": DIMENSIONLESS, "applies_to": BOTH, "constraints": [],
     "description": "Finite replacement for the infinite hard-core repulsion: f(r) = -hard_core_strength for r <= 2*r0."},
    {"name": "simulation.overlap_iterations", "type": "integer", "default": 3,
     "range": {"min": 0, "max": None}, "explored_range": None, "unit": "passes",
     "applies_to": BOTH, "constraints": [],
     "description": "Overlap-projection passes after each update. Changing it changes the numerical trajectory; 3 does not guarantee zero overlaps in crowded states (see diagnostics)."},
    {"name": "simulation.seed", "type": "integer", "default": "chosen-and-recorded-if-omitted",
     "range": {"min": 0, "max": None}, "explored_range": None, "unit": None,
     "applies_to": BOTH, "constraints": [],
     "description": ("Seed for numpy.random.default_rng; controls initial positions, headings, noise, "
                     "and (two species, via seed+7919) species labels. All randomness flows from it. "
                     "If omitted, `run` picks one, records it, and reports it on stderr. "
                     "(The legacy `simulate` command and SimulationConfig default to 0.)")},
    {"name": "simulation.use_cell_list", "type": "boolean", "default": False, "unit": None,
     "applies_to": BOTH, "constraints": [],
     "description": "Use the linked-cell neighbor accelerator (spp_unified/neighbors.py) instead of dense all-pairs. Its docstring states it finds exactly the same within-cutoff pairs; it falls back to the dense path for small n."},

    # --- recording ---
    {"name": "state_every", "type": "integer", "default": 8,
     "range": {"min": 1, "max": None}, "explored_range": None, "unit": "updates",
     "applies_to": BOTH, "constraints": [],
     "description": "Record positions and headings every state_every updates (plus the last). Descriptors use the last 10 recorded frames, so this changes the descriptor."},
]

# Bookkeeping key, not a model parameter.
SCHEMA_KEY = {"name": "schema", "type": "integer", "default": CONFIG_SCHEMA_VERSION,
              "options": [CONFIG_SCHEMA_VERSION], "unit": None, "applies_to": BOTH,
              "constraints": [], "description": "Configuration format version."}

# ---------------------------------------------------------------------------
# Cross-parameter constraints. ``severity`` error -> exit 2, warning -> reported only.
# ---------------------------------------------------------------------------
CONSTRAINTS = {
    "sample_start_before_steps": {
        "expression": "simulation.sample_start < simulation.steps",
        "severity": "error", "fields": ["simulation.sample_start", "simulation.steps"],
        "description": "Sampling must start before the run ends."},
    "box_exceeds_diameter": {
        "expression": "simulation.S > 2 * simulation.r0",
        "severity": "error", "fields": ["simulation.S", "simulation.r0"],
        "description": "The box must be larger than one particle diameter."},
    "particle_placement": {
        "expression": "n particles fit on the initial grid with spacing >= 2.02*r0",
        "severity": "error", "fields": ["parameters.rho", "simulation.S", "simulation.r0"],
        "description": "Initial placement (geometry.initialize_positions) raises if rho is too high for the box."},
    "r1_within_half_box": {
        "expression": "parameters.interaction_range_ratio * simulation.r0 <= simulation.S / 2",
        "severity": "warning", "fields": ["parameters.interaction_range_ratio", "simulation.S", "simulation.r0"],
        "description": "Clustering's periodic normalization and clean minimum-image geometry assume r1 <= S/2 (docs/NUMERICS.md). Runs still execute."},
    "adhesion_clip": {
        "expression": "abs(parameters.adhesion) <= 0.95",
        "severity": "warning", "fields": ["parameters.adhesion"],
        "description": "beta_from_adhesion clips B to [-0.95, 0.95]; larger magnitudes are silently equivalent to +/-0.95."},
    "mixture_requires_two": {
        "expression": "mixture block only when kind == 'two'",
        "severity": "error", "fields": ["mixture"],
        "description": "The single model ignores mixture parameters; giving them is treated as a mistake."},
}

# ---------------------------------------------------------------------------
# Outputs.
# ---------------------------------------------------------------------------
SUMMARY_OUTPUTS = [
    {"name": "n", "unit": "particles", "shape": "scalar", "description": "Particle count (from rho, S, r0)."},
    {"name": "phi", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "Polar alignment |mean heading vector|, averaged over sampled steps. ~1 flock, ~0 disordered."},
    {"name": "nematic", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "Nematic order S2 = |mean exp(2i theta)|, averaged over sampled steps."},
    {"name": "milling", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "Milling m: normalized angular momentum about the (circular) group center, averaged. ~1 for a rotating mill."},
    {"name": "clustering", "unit": "ratio to ideal gas", "shape": "scalar",
     "description": "Clustering Gamma: mean neighbors within r1 divided by the ideal-gas expectation. 1 = uniform, >1 clustered."},
    {"name": "segregation", "unit": DIMENSIONLESS, "shape": "scalar", "range": [-1, 1],
     "description": "Species segregation within r1 relative to random mixing (two species only; 0 for single)."},
    {"name": "drift", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "Net center-of-mass transport per unit commanded path length (uses commanded displacement before overlap correction)."},
    {"name": "omega", "unit": "radian per simulation time unit", "shape": "scalar",
     "description": "Signed rotation rate of the global heading (polarization-weighted fit)."},
    {"name": "abs_omega", "unit": "radian per simulation time unit", "shape": "scalar",
     "description": "|omega|."},
    {"name": "rotation_r2", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "R^2 of the linear fit of unwrapped global heading angle; persistence of rotation."},
    {"name": "chirality", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "Chirality score clip(|omega|/0.05,0,1) * rotation_r2 * mean(phi)."},
    {"name": "temporal_stability", "unit": DIMENSIONLESS, "shape": "scalar", "range": [0, 1],
     "description": "1 - mean of the standard deviations of phi, milling, and scaled clustering over the sample window."},
]
DIAGNOSTIC_OUTPUTS = [
    {"name": "final_minimum_separation", "unit": LENGTH, "shape": "scalar",
     "description": "Smallest pair distance in the final frame."},
    {"name": "diameter", "unit": LENGTH, "shape": "scalar", "description": "Hard-core diameter 2*r0."},
    {"name": "final_overlapping_pairs", "unit": "pairs", "shape": "scalar",
     "description": "Pairs closer than 2*r0 in the final frame (residual overlaps after projection)."},
]
SERIES_OUTPUTS = [
    {"name": "phi", "unit": DIMENSIONLESS, "description": "Polar alignment at each sampled step."},
    {"name": "nematic", "unit": DIMENSIONLESS, "description": "Nematic order at each sampled step."},
    {"name": "milling", "unit": DIMENSIONLESS, "description": "Milling at each sampled step."},
    {"name": "clustering", "unit": "ratio to ideal gas", "description": "Clustering at each sampled step."},
    {"name": "segregation", "unit": DIMENSIONLESS, "description": "Species segregation at each sampled step (zeros for single)."},
    {"name": "com_x", "unit": LENGTH, "description": "Unwrapped center of mass x (commanded displacement, before overlap correction)."},
    {"name": "com_y", "unit": LENGTH, "description": "Unwrapped center of mass y."},
]
TRAJECTORY_ARRAYS = [
    {"name": "positions", "unit": LENGTH, "shape": "[frames, n, 2]", "description": "Recorded positions (wrapped into the box)."},
    {"name": "headings", "unit": "radian", "shape": "[frames, n]", "description": "Recorded headings in [-pi, pi)."},
    {"name": "species", "unit": None, "shape": "[n]", "description": "Species label 0/1 (all 0 for single)."},
    {"name": "descriptor", "unit": DIMENSIONLESS, "shape": "[51] single, [107] two",
     "description": "Learned-representation input descriptor of the last 10 recorded frames."},
    {"name": "recorded_steps", "unit": "updates", "shape": "[frames]", "description": "One-based update count of each recorded frame."},
    {"name": "timeseries_<name>", "unit": "see series outputs", "shape": "[samples] (com: [samples,2], theta: [samples,n])",
     "description": "Raw sampled order parameters, as in timeseries.csv; timeseries_theta holds headings at each sample."},
]

INTERVENTIONS: list[dict] = []
INTERVENTIONS_NOTE = ("The simulator has no time-dependent interventions: all parameters are constant "
                      "during a run. The taxis cue (gradient_*) is a constant field, not an event.")

FIXED_CONSTANTS = [
    {"name": "alignment_self_weight", "value": 1.0, "where": "spp_unified/model.py",
     "description": "Focal particle's own heading is included with unit weight in alignment."},
    {"name": "same_species_alignment_weight", "value": 1.0, "where": "TwoSpeciesParams.matrices"},
    {"name": "same_species_additive_force", "value": 0.0, "where": "TwoSpeciesParams.matrices"},
    {"name": "adhesion_clip", "value": [-0.95, 0.95], "where": "model.beta_from_adhesion"},
    {"name": "tan_clip", "value": [-8.0, 8.0], "where": "model.beta_from_adhesion"},
    {"name": "min_particles", "value": 8, "where": "model._simulate"},
    {"name": "initial_min_separation", "value": "2.02*r0", "where": "geometry.initialize_positions"},
    {"name": "species_label_seed_offset", "value": 7919, "where": "model.simulate_two_species"},
    {"name": "chirality_rate_scale", "value": 0.05, "where": "model._simulate"},
    {"name": "descriptor_max_frames", "value": 10, "where": "learned.behavior_descriptor"},
    {"name": "descriptor_nbins", "value": {"single": 12, "two_extra": 8}, "where": "learned"},
]


def parameter_index() -> dict[str, dict]:
    return {p["name"]: p for p in PARAMETERS}


def parameters_for(model: str) -> list[dict]:
    return [p for p in PARAMETERS if model in p["applies_to"]]


def output_names() -> list[str]:
    return [o["name"] for o in SUMMARY_OUTPUTS]
