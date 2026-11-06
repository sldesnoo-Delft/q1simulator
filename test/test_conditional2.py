"""
Conditional playback on a QCM using a thresholded acquisition trigger of a QRM.

Every scenario runs with and without trigger. The simulated acquisition value
is the acquisition time in ms (< 0.2), so threshold -0.5 always triggers and
threshold 0.9 never triggers.

Simulator only: checks sequencer.get_output().
"""
import numpy as np
import pytest

from q1_helpers import lsb, render_path


V_MAX = 2.5  # QCM
N_REP = 200
THRESHOLDS = {True: -0.5, False: 0.9}

RAMP_LIMITS = [
    (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0),
    (1.0, 0.5), (0.5, 0.0), (0.0, -0.5), (-0.5, -1.0),
]
RAMPS = [0.5 * np.linspace(start, stop, 60) for start, stop in RAMP_LIMITS]
WAVEFORMS = {f"ramp{i}": {"data": ramp.tolist(), "index": i} for i, ramp in enumerate(RAMPS)}
ACQUISITIONS = {"acq0": {"num_bins": 1, "index": 0}}


def qrm_program(wait):
    return f"""
move {N_REP},R1
wait_sync 100
start:
  acquire 0, 0, 100
  wait {wait}
loop R1,@start
stop
"""


QCM_IF_ELSE_PLAY = f"""
move {N_REP},R1
wait_sync 100
start:
  set_latch_en 1, 100
  set_awg_gain 32767,32767
  set_mrk 15
  play 0,0,100
  set_mrk 0
  play 3,3,100
  wait 92 # subtract 8 for next conditional

  # set_cond enable, 0b0001, OR, else_wait
  set_cond 1, 1, 0, 4
  wait 8 # 4ns * number of statements including this wait
  play 1,1,92 # subtract 8 ns for next conditional
  # duration of else: 8 ns

  # set_cond enable, 0b0001, NOR, else_wait
  set_cond 1, 1, 1, 4
  play 6,6,80
  wait 20

  # disable conditional
  set_cond 0, 0, 0, 0
  play 3,3,96
  latch_rst 4
loop R1,@start
stop
"""

QCM_IF_ELSE_OFFSET = f"""
move {N_REP},R1
wait_sync 100
start:
  upd_param 4
  set_latch_en 1, 96
  set_awg_gain 32767,32767
  set_mrk 15
  play 0,0,100
  set_mrk 0
  play 3,3,100
  wait 92 # subtract 8 for next conditional

  # set_cond enable, 0b0001, OR, else_wait
  set_cond 1, 1, 0, 4
  set_awg_offs 16384,16384
  wait 8 # 4ns * number of statements including this wait
  upd_param 92 # subtract 8 ns for next conditional
  # duration of else: 8 ns

  # set_cond enable, 0b0001, NOR, else_wait
  set_cond 1, 1, 1, 4
  set_awg_offs -16384,-16384
  upd_param 80
  wait 20

  # disable conditional
  set_cond 0, 0, 0, 0
  play 3,3,96
  latch_rst 4
  set_awg_offs 0,0
loop R1,@start
set_awg_offs 0,0
upd_param 4
stop
"""

TRIGGER_DELAY = 248
QCM_CONDITION_EVALUATION = f"""
move {N_REP},R1
wait_sync 100
start:
  set_latch_en 1, 100
  set_awg_gain 32767,32767
  set_mrk 15
  play 1,1,100
  set_mrk 0
  play 3,3,60
  wait {TRIGGER_DELAY-160}

  # set_cond enable, 0b0001, OR, else_wait
  set_cond 1, 1, 0, 4
""" + "  play 1,1,8\n" * 10 + """
  wait 16 # 10 statements, subtract 4 ns for else: 100-80-4

  # set_cond enable, 0b0001, NOR, else_wait
  set_cond 1, 1, 1, 4
  wait 76 # 100-6*4

  # disable conditional
  set_cond 0, 0, 0, 0
  play 3,3,96
  latch_rst 4
loop R1,@start
set_awg_offs 0,0
upd_param 4
stop
"""


def run(cluster, qrm_program, qcm_program, trigger: bool) -> np.ndarray:
    """Runs QRM sequencer 0 emitting trigger 1 and QCM sequencer 1. Returns QCM output path 1."""
    qcm, qrm = cluster.module1, cluster.module2
    qrm_seq = qrm.sequencers[0]
    qrm_seq.connect_out0("I")
    qrm_seq.integration_length_acq(100)
    qrm_seq.thresholded_acq_trigger_en(True)
    qrm_seq.thresholded_acq_trigger_address(1)
    qrm_seq.thresholded_acq_threshold(THRESHOLDS[trigger])
    qrm_seq.thresholded_acq_trigger_invert(False)
    qrm_seq.sequence({"program": qrm_program, "waveforms": {}, "weights": {}, "acquisitions": ACQUISITIONS})

    qcm_seq = qcm.sequencers[1]
    qcm_seq.connect_out1("Q")
    qcm_seq.sequence({"program": qcm_program, "waveforms": WAVEFORMS, "weights": {}, "acquisitions": {}})

    for module, seq_nr in [(qrm, 0), (qcm, 1)]:
        module.sequencers[seq_nr].sync_en(True)
        module.arm_sequencer(seq_nr)
    cluster.start_sequencer()
    for module, seq_nr in [(qrm, 0), (qcm, 1)]:
        assert module.get_sequencer_status(seq_nr, 1).err_flags == []

    return qcm_seq.get_output()["Q"].data


def assert_output(data, expected):
    np.testing.assert_allclose(data, V_MAX * expected, rtol=0, atol=3 * lsb(V_MAX))


@pytest.mark.parametrize("trigger", [False, True], ids=["no-trigger", "trigger"])
def test_if_else_play(sim_cluster, trigger):
    data = run(sim_cluster({1: "QCM", 2: "QRM"}), qrm_program(500), QCM_IF_ELSE_PLAY, trigger)

    # Both branches take 100 ns. Repetition takes 600 ns, like the QRM.
    plays = []
    for t in range(0, N_REP * 600, 600):
        conditional = RAMPS[1] if trigger else RAMPS[6]
        plays += [(t + 100, RAMPS[0], 1.0), (t + 200, RAMPS[3], 1.0),
                  (t + 400, conditional, 1.0), (t + 500, RAMPS[3], 1.0)]
    assert_output(data, render_path(N_REP * 600, plays))


@pytest.mark.parametrize("trigger", [False, True], ids=["no-trigger", "trigger"])
def test_if_else_offset(sim_cluster, trigger):
    data = run(sim_cluster({1: "QCM", 2: "QRM"}), qrm_program(500), QCM_IF_ELSE_OFFSET, trigger)

    duration = N_REP * 600 + 4
    plays = []
    offsets = np.zeros(duration)
    for t in range(0, N_REP * 600, 600):
        plays += [(t + 100, RAMPS[0], 1.0), (t + 200, RAMPS[3], 1.0), (t + 500, RAMPS[3], 1.0)]
        if trigger:
            offsets[t + 400:t + 500] = 0.5
            # The set_awg_offs -16384 of the skipped branch is still latched
            # and applied by the play at 500 ns.
            offsets[t + 500:t + 600] = -0.5
        else:
            offsets[t + 400:t + 600] = -0.5
    assert_output(data, render_path(duration, plays) + offsets)


def test_condition_evaluation_no_trigger(sim_cluster):
    data = run(sim_cluster({1: "QCM", 2: "QRM"}), qrm_program(TRIGGER_DELAY + 200),
               QCM_CONDITION_EVALUATION, trigger=False)

    # Without trigger the 10 plays are skipped: 10 x else_wait 4 + 4 for wait.
    # Repetition takes 568 ns.
    plays = []
    for t in range(0, N_REP * 568, 568):
        plays += [(t + 100, RAMPS[1], 1.0), (t + 200, RAMPS[3], 1.0), (t + 468, RAMPS[3], 1.0)]
    assert_output(data, render_path(N_REP * 568 + 4, plays))


def test_condition_evaluation_trigger(sim_cluster):
    data = run(sim_cluster({1: "QCM", 2: "QRM"}), qrm_program(TRIGGER_DELAY + 200),
               QCM_CONDITION_EVALUATION, trigger=True)

    # Only the first repetition is checked. The repetition times of QRM and QCM differ,
    # so the trigger moves with respect to the conditional plays in the next repetitions.
    # Trigger of acquisition [0, 100) arrives at 350 ns: first play at 348 ns is skipped (4 ns),
    # the next 9 plays start at 352 ns. The last one continues till the play at 444 ns.
    plays = [(100, RAMPS[1], 1.0), (200, RAMPS[3], 1.0)]
    plays += [(352 + 8 * i, RAMPS[1], 1.0) for i in range(9)]
    plays += [(444, RAMPS[3], 1.0)]
    assert_output(data[:544], render_path(544, plays))
