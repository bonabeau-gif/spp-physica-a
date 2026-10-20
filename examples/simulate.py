"""Small API example. Run from the repository after installation."""
from spp_unified import SingleParams, SimulationConfig, simulate_single
from spp_unified.learned import behavior_descriptor

parameters = SingleParams(rho=0.1, noise=0.1, force_strength=0.0)
config = SimulationConfig(steps=160, sample_start=80, seed=42)
result = simulate_single(parameters, config, record_states=True, state_every=8)
print(result.summary)
print("Descriptor shape:", behavior_descriptor(result.states, config).shape)

