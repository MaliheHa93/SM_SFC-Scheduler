from __future__ import annotations

from pathlib import Path
import unittest

from stg_ddqn.config import load_config
from stg_ddqn.environment import STGEnvironment
from stg_ddqn.neural import TORCH_AVAILABLE, STGDDQNPolicy
from stg_ddqn.requests import RequestGenerator


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in the minimal runtime")
class NeuralShapeTests(unittest.TestCase):
    def test_masked_network_returns_a_valid_action(self):
        cfg = load_config(ROOT / "configs" / "smoke.yaml")
        environment = STGEnvironment(cfg, 11)
        environment.vehicles = environment.mobility.state_at(0.0)
        environment.state_history.record(0.0, environment.vehicles)
        request = environment.request_generator._sample(
            0.0, sorted(environment.vehicles)[0], tracked=True
        )
        vehicle = environment.vehicles[request.vehicle_id]
        access = environment.topology.access_node(vehicle.x_m, vehicle.y_m)
        candidates = environment.candidate_generator.build(request, None, access, 0.0)
        observation = environment.state_history.build(
            request, None, candidates, 0.0, 0.0, False, 1
        )
        policy = STGDDQNPolicy(cfg, seed=11)
        action = policy.select(observation, training=False)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, len(candidates))


if __name__ == "__main__":
    unittest.main()

