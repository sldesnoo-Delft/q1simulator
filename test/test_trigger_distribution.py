"""
Distribution of acquisition triggers between sequencers with conditional playback.

QRM sequencer 0 emits trigger 1 and QRM sequencer 1 emits trigger 2.
The QCM sequencer and both QRM sequencers play conditionally on a set of triggers.
The trigger value is forced with the simulator configuration `acq_trigger_value`.

The simulator skips the repetitions of loops to label `_start`, so 1 repetition is rendered.

Simulator only: checks sequencer.get_output().
"""
import numpy as np
import pytest

from q1_helpers import lsb, render_path


N_REP = 20
QCM_V_MAX = 2.5
QRM_V_MAX = 0.5
EMITTED_TRIGGERS = {0: 1, 1: 2}  # QRM sequencer -> trigger address

RAMP_LIMITS = [
    (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0),
    (1.0, 0.5), (0.5, 0.0), (0.0, -0.5), (-0.5, -1.0),
]
RAMPS = [0.5 * np.linspace(start, stop, 60) for start, stop in RAMP_LIMITS]
WAVEFORMS = {f"ramp{i}": {"data": ramp.tolist(), "index": i} for i, ramp in enumerate(RAMPS)}
ACQUISITIONS = {"acq0": {"num_bins": 1, "index": 0}}


def trigger_mask(used_triggers):
    mask = 0
    for addr in used_triggers:
        mask |= 1 << (addr - 1)
    return mask


def qcm_program(used_triggers):
    mask = trigger_mask(used_triggers)
    return f"""
    move {N_REP},R1
    wait_sync 100
    _start:
      set_latch_en 1, 100
      set_awg_gain 32767,32767
      set_mrk 1
      play 0,0,100
      set_mrk 0
      play 3,3,100
      wait 92 # subtract 8 for next conditional

      # set_cond enable, mask, OR, else_wait
      set_cond 1, {mask}, 0, 4
      wait 8 # 4ns * number of statements including this wait
      play 1,1,92 # subtract 8 ns for next conditional
      # duration of else: 8 ns

      # set_cond enable, mask, NOR, else_wait
      set_cond 1, {mask}, 1, 4
      play 6,6,80
      wait 20

      # disable conditional
      set_cond 0, 0, 0, 0
      play 3,3,96
      latch_rst 4
    loop R1,@_start
    stop
    """


def qrm_program(used_triggers):
    mask = trigger_mask(used_triggers)
    return f"""
    move {N_REP},R1
    wait_sync 100
    _start:
      set_latch_en 1, 4
      acquire 0, 0, 96
      set_awg_gain 32767,32767
      play 0,0,100
      play 3,3,100
      wait 92 # subtract 8 for next conditional

      # set_cond enable, mask, OR, else_wait
      set_cond 1, {mask}, 0, 4
      wait 8 # 4ns * number of statements including this wait
      play 1,1,92 # subtract 8 ns for next conditional
      # duration of else: 8 ns

      # set_cond enable, mask, NOR, else_wait
      set_cond 1, {mask}, 1, 4
      play 6,6,80
      wait 20

      # disable conditional
      set_cond 0, 0, 0, 0
      play 3,3,96
      latch_rst 4
    loop R1,@_start
    stop
    """


# (triggers used by QCM sequencer 0, by QRM sequencer 0, by QRM sequencer 1)
SCENARIOS = [
    ([1, 2], [], []),
    ([1], [2], []),
    ([1], [2], [2]),
    ([2], [1, 2], [2]),
    # Dependencies on triggers that are not emitted.
    ([1, 2, 3], [1, 2], [2, 9]),
    # QRM sequencers depend on each other's trigger.
    ([], [2], [1, 2]),
]


def scenario_id(scenario):
    return "-".join(f"{name}{''.join(map(str, used)) or 'none'}"
                    for name, used in zip(["qcm", "qrm0_", "qrm1_"], scenario))


def expected_output(condition: bool) -> np.ndarray:
    """Output of 1 repetition: ramp1 if condition is true, else ramp6."""
    conditional = RAMPS[1] if condition else RAMPS[6]
    return render_path(600, [(100, RAMPS[0], 1.0), (200, RAMPS[3], 1.0),
                             (400, conditional, 1.0), (500, RAMPS[3], 1.0)])


@pytest.mark.parametrize("trigger_value", [0, 1])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario_id(s) for s in SCENARIOS])
def test_trigger_distribution(sim_cluster, scenario, trigger_value):
    qcm_used, qrm0_used, qrm1_used = scenario
    cluster = sim_cluster({1: "QCM", 2: "QRM"})
    qcm, qrm = cluster.module1, cluster.module2

    # (sequencer, used triggers, output path, v_max)
    sequencers = [(qcm.sequencers[0], qcm_used, "I", QCM_V_MAX)]
    qcm.sequencers[0].connect_out0("I")
    qcm.sequencers[0].sequence({"program": qcm_program(qcm_used), "waveforms": WAVEFORMS,
                                "weights": {}, "acquisitions": {}})
    for seq_nr, used in [(0, qrm0_used), (1, qrm1_used)]:
        seq = qrm.sequencers[seq_nr]
        path = "IQ"[seq_nr]
        seq.parameters[f"connect_out{seq_nr}"].set(path)
        seq.integration_length_acq(100)
        seq.thresholded_acq_trigger_en(True)
        seq.thresholded_acq_trigger_address(EMITTED_TRIGGERS[seq_nr])
        seq.thresholded_acq_threshold(0.0)
        seq.thresholded_acq_trigger_invert(False)
        seq.sequence({"program": qrm_program(used), "waveforms": WAVEFORMS,
                      "weights": {}, "acquisitions": ACQUISITIONS})
        sequencers.append((seq, used, path, QRM_V_MAX))

    for seq, *_ in sequencers:
        seq.sync_en(True)
        seq.arm_sequencer()
    cluster.config("acq_trigger_value", trigger_value)
    cluster.start_sequencer()

    emitted = set(EMITTED_TRIGGERS.values()) if trigger_value else set()
    for seq, used, path, v_max in sequencers:
        assert seq.get_sequencer_status(timeout=1).err_flags == [], seq.name
        # Triggers of acquisition at 4 ns arrive at 354 ns, before set_cond at 392 ns.
        condition = bool(set(used) & emitted)
        np.testing.assert_allclose(
            seq.get_output()[path].data, v_max * expected_output(condition),
            rtol=0, atol=3 * lsb(v_max), err_msg=f"{seq.name}: condition {condition}")
