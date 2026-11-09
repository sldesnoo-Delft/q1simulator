import itertools
import logging

import pytest
from qblox_instruments import Cluster, ClusterType

from q1simulator import Q1Plotter

PROGRAM = """
       wait 20
       play 0, 0, 100
       stop
"""

SEQUENCE = {
    "waveforms": {"ones80": {"data": [1.0] * 80, "index": 0}},
    "weights": {},
    "acquisitions": {},
    "program": PROGRAM,
}

SUPPORTED_MODULES = {
    2: ClusterType.CLUSTER_QCM,
    4: ClusterType.CLUSTER_QRM,
    6: ClusterType.CLUSTER_QCM_RF,
    8: ClusterType.CLUSTER_QRM_RF,
}

UNKNOWN_MODULES = {
    10: ClusterType.CLUSTER_QTM,
    12: ClusterType.CLUSTER_QSM,
}

_name_counter = itertools.count()


@pytest.fixture
def dummy_cluster():
    dummy_cfg = {**SUPPORTED_MODULES, **UNKNOWN_MODULES}
    cluster = Cluster(f"qbi_dummy{next(_name_counter)}", identifier=None, dummy_cfg=dummy_cfg)
    yield cluster
    cluster.close()


def test_q1plotter_not_raising_for_any_existing_modules(dummy_cluster, caplog):
    """
    Q1Plotter on Qblox dummy cluster: Q1Plotter is able to produce output if all
    available modules are in the cluster. Q1Simulator modules produce output,
    unknown modules raise warnings.
    """
    for slot in SUPPORTED_MODULES:
        seq = dummy_cluster.modules[slot - 1].sequencers[0]
        seq.sync_en(True)
        seq.sequence(SEQUENCE)

    with caplog.at_level(logging.WARNING, logger="q1simulator.q1plotter"):
        plotter = Q1Plotter(dummy_cluster)

    sim_modules = plotter._simulator.modules
    simulated_slots = [
        slot for slot, module in enumerate(sim_modules, 1) if module.present()
    ]
    assert simulated_slots == list(SUPPORTED_MODULES)

    for slot in SUPPORTED_MODULES:
        output = sim_modules[slot - 1].sequencers[0].get_output()
        assert output, f"no output for slot {slot}"
        assert all(len(channel.data) > 0 for channel in output.values()), (
            f"empty output for slot {slot}"
        )

    for slot in UNKNOWN_MODULES:
        assert (
            "unknown module type" in caplog.text and f"in slot {slot}" in caplog.text
        )
    for slot in SUPPORTED_MODULES:
        assert f"in slot {slot}" not in caplog.text
