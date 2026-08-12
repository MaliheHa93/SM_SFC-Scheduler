from __future__ import annotations

import numpy as np

from .types import SFCRequest, VNFSpec


class RequestGenerator:
    def __init__(self, cfg: dict, epoch_s: float, rng_arrivals: np.random.Generator,
                 rng_requests: np.random.Generator):
        self.cfg = cfg
        self.epoch_s = float(epoch_s)
        self.rng_arrivals = rng_arrivals
        self.rng_requests = rng_requests
        self.next_request_id = 0

    def generate(self, time_s: float, active_vehicle_ids: list[int],
                 tracked: bool) -> list[SFCRequest]:
        arrival_rate = float(self.cfg["arrival_rate_req_s"])
        count = int(self.rng_arrivals.poisson(arrival_rate * self.epoch_s))
        requests: list[SFCRequest] = []
        if not active_vehicle_ids:
            return requests
        for _ in range(count):
            vehicle_id = int(self.rng_requests.choice(active_vehicle_ids))
            requests.append(self._sample(time_s, vehicle_id, tracked))
        return requests

    def _sample(self, time_s: float, vehicle_id: int, tracked: bool) -> SFCRequest:
        rng = self.rng_requests
        length_low, length_high = self.cfg["sfc_length"]
        length = int(rng.integers(int(length_low), int(length_high) + 1))
        cpu_low, cpu_high = self.cfg["cpu_requirement"]
        work_low, work_high = self.cfg["workload_mi"]
        image_low, image_high = self.cfg["vnf_image_mb"]
        state_mb = float(self.cfg["runtime_state_mb"])
        vnfs = tuple(
            VNFSpec(
                cpu=float(rng.uniform(cpu_low, cpu_high)),
                workload_mi=float(rng.uniform(work_low, work_high)),
                image_mb=float(rng.uniform(image_low, image_high)),
                state_mb=state_mb,
            )
            for _ in range(length)
        )
        bandwidth_low, bandwidth_high = self.cfg["bandwidth_mb_s"]
        latency_low, latency_high = self.cfg["latency_requirement_ms"]
        lifetime_low, lifetime_high = self.cfg["service_lifetime_s"]
        request = SFCRequest(
            request_id=self.next_request_id,
            vehicle_id=vehicle_id,
            arrival_s=float(time_s),
            lifetime_s=float(rng.uniform(lifetime_low, lifetime_high)),
            bandwidth_mb_s=float(rng.uniform(bandwidth_low, bandwidth_high)),
            latency_requirement_ms=float(rng.uniform(latency_low, latency_high)),
            vnfs=vnfs,
            tracked=bool(tracked),
        )
        self.next_request_id += 1
        return request

