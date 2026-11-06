"""
Shared pytest fixtures.

Tests using the `backend` or `qcm_qrm_backend` fixture run on the simulator and,
when an IP address is given, also on a Qblox cluster:

    pytest test/                                      # simulator only
    pytest test/ --hw-ip 192.168.0.2                  # simulator and hardware
    pytest test/ --hw-ip 192.168.0.2 --hw-slot 4 --hw-qcm-slot 2 --hw-qrm-slot 6
"""
import itertools
import time
from dataclasses import dataclass
from typing import Any

import pytest


_name_counter = itertools.count()


def pytest_addoption(parser):
    parser.addoption("--hw-ip", default=None,
                     help="IP address of Qblox cluster. Hardware tests are skipped if not given.")
    parser.addoption("--hw-slot", type=int, default=2,
                     help="Slot of the module used for single module hardware tests (default: 2).")
    parser.addoption("--hw-qcm-slot", type=int, default=2,
                     help="Slot of the QCM used for hardware tests with QCM and QRM (default: 2).")
    parser.addoption("--hw-qrm-slot", type=int, default=6,
                     help="Slot of the QRM used for hardware tests with QCM and QRM (default: 6).")


def _wait_until_stopped(sequencers, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(seq.q1core.running for seq in sequencers):
            return True
        time.sleep(0.05)
    return False


def _close_cluster(cluster, is_simulator: bool):
    cluster.stop_sequencer()
    if is_simulator:
        sequencers = [
            seq
            for module in cluster.get_connected_modules().values()
            for seq in module.sequencers
        ]
        assert _wait_until_stopped(sequencers, timeout=5), "sequencer threads still running after stop"
    cluster.close()


def _connect_hw_cluster(request):
    ip = request.config.getoption("--hw-ip")
    if ip is None:
        pytest.skip("hardware test: no --hw-ip given")
    from qblox_instruments import Cluster

    return Cluster(f"q1hw_test{next(_name_counter)}", ip)


@dataclass
class Backend:
    """Cluster with 1 module used for test."""
    name: str
    cluster: Any
    slot: int

    @property
    def module(self):
        return getattr(self.cluster, f"module{self.slot}")

    @property
    def is_simulator(self) -> bool:
        return self.name == "sim"


@pytest.fixture(scope="module", params=["sim", "hw"])
def backend(request):
    if request.param == "sim":
        from q1simulator import Cluster as SimCluster

        slot = 2
        cluster = SimCluster(f"q1sim_test{next(_name_counter)}", {slot: "QRM"}, isa_version=(2, 0))
    else:
        cluster = _connect_hw_cluster(request)
        slot = request.config.getoption("--hw-slot")

    backend = Backend(request.param, cluster, slot)
    yield backend
    _close_cluster(cluster, backend.is_simulator)


@dataclass
class QcmQrmBackend:
    """Cluster with a QCM and a QRM used for test."""
    name: str
    cluster: Any
    qcm_slot: int
    qrm_slot: int

    @property
    def qcm(self):
        return getattr(self.cluster, f"module{self.qcm_slot}")

    @property
    def qrm(self):
        return getattr(self.cluster, f"module{self.qrm_slot}")

    @property
    def is_simulator(self) -> bool:
        return self.name == "sim"


@pytest.fixture(scope="module", params=["sim", "hw"])
def qcm_qrm_backend(request):
    if request.param == "sim":
        from q1simulator import Cluster as SimCluster

        qcm_slot, qrm_slot = 2, 6
        cluster = SimCluster(f"q1sim_test{next(_name_counter)}", {qcm_slot: "QCM", qrm_slot: "QRM"},
                             isa_version=(2, 0))
    else:
        cluster = _connect_hw_cluster(request)
        qcm_slot = request.config.getoption("--hw-qcm-slot")
        qrm_slot = request.config.getoption("--hw-qrm-slot")

    backend = QcmQrmBackend(request.param, cluster, qcm_slot, qrm_slot)
    yield backend
    _close_cluster(cluster, backend.is_simulator)


@pytest.fixture
def sim_cluster():
    """Factory for simulated clusters. Clusters are stopped and closed after the test.

    Usage: cluster = sim_cluster({1: "QCM", 2: "QRM"})
    """
    from q1simulator import Cluster as SimCluster

    clusters = []

    def create(modules: dict[int, str]):
        cluster = SimCluster(f"q1sim_test{next(_name_counter)}", modules, isa_version=(2, 0))
        clusters.append(cluster)
        return cluster

    yield create
    for cluster in clusters:
        _close_cluster(cluster, is_simulator=True)
