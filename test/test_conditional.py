"""
Conditional playback with set_cond on a trigger set with #Q1Sim:sim_trigger.

Simulator only: checks sequencer.get_output().
"""
import numpy as np
import pytest

from q1_helpers import lsb, render_path


V_MAX = 0.5  # QRM
# mod_en_awg with NCO frequency 0 Hz scales output by 1/sqrt(2).
SCALE = 1 / np.sqrt(2)

GAUSS80 = [
    0.00021073734644620426, 0.00032175475915178565, 0.000485954999850973, 0.0007260302794298448,
    0.0010730031959653244, 0.0015686819551156438, 0.0022685919639534573, 0.003245379444956216,
    0.004592637591638869, 0.0064290451339909255, 0.008902631069565252, 0.012194889831651198,
    0.01652437299454718, 0.022149284470989822, 0.02936851692650569, 0.038520501607367155,
    0.04997921782427386, 0.06414673848332593, 0.08144178947697502, 0.10228398474484361,
    0.1270736700907338, 0.156167662882968, 0.1898515957906482, 0.22831003268047262,
    0.27159598392359746, 0.31960185739844454, 0.3720341859374524, 0.42839461801732714,
    0.48796960000609346, 0.5498308842335518, 0.6128484583099875, 0.6757167248755714,
    0.7369938133110109, 0.7951528488218665, 0.8486429340866658, 0.8959566209925681,
    0.9356998720224928, 0.9666600271750343, 0.9878671723140003, 0.9986445826034864,
    0.9986445826034864, 0.9878671723140003, 0.9666600271750343, 0.9356998720224928,
    0.8959566209925681, 0.8486429340866658, 0.7951528488218665, 0.7369938133110109,
    0.6757167248755714, 0.6128484583099875, 0.5498308842335518, 0.48796960000609346,
    0.42839461801732714, 0.3720341859374524, 0.31960185739844454, 0.27159598392359746,
    0.22831003268047262, 0.1898515957906482, 0.156167662882968, 0.1270736700907338,
    0.10228398474484361, 0.08144178947697502, 0.06414673848332593, 0.04997921782427386,
    0.038520501607367155, 0.02936851692650569, 0.022149284470989822, 0.01652437299454718,
    0.012194889831651198, 0.008902631069565252, 0.0064290451339909255, 0.004592637591638869,
    0.003245379444956216, 0.0022685919639534573, 0.0015686819551156438, 0.0010730031959653244,
    0.0007260302794298448, 0.000485954999850973, 0.00032175475915178565, 0.00021073734644620426,
]
ZERO80 = [0.0] * 80

WAVEFORMS = {
    "gauss80": {"data": GAUSS80, "index": 0},
    "zero80": {"data": ZERO80, "index": 1},
}

# Trigger 1 is set. Triggers 2 and 3 are not.
PROGRAM_IF_ELSE = """
wait_sync 100
set_latch_en 1,4
#Q1Sim:sim_trigger 1, 1

play 0,0,92 # subtract 8 for next conditional

set_cond 1, 1, 0, 4
wait 8 # 4ns * number of statements including this wait
play 0,1,72 # subtract 8 ns for next conditional
# duration of else: 8 ns

set_cond 1, 2, 0, 4
play 1,0,60
wait 20

set_cond 0, 0, 0, 0
play 1,1,100
stop
"""

# Expected: (duration, plays path 0, plays path 1). Plays are (time, waveform, gain).
EXPECTED_IF_ELSE = (
    284,
    # play 0,0 at 4; cond 1 true: play 0,1 at 104; cond 2 false: 2 x else_wait 4; play 1,1 at 184.
    [(4, GAUSS80, 1.0), (104, GAUSS80, 1.0), (184, ZERO80, 1.0)],
    [(4, GAUSS80, 1.0), (104, ZERO80, 1.0), (184, ZERO80, 1.0)],
)

PROGRAM_IF_ELIF_ELSE = """
wait_sync 100
set_awg_gain 32000,32000
reset_ph
set_latch_en 1,4
#Q1Sim:sim_trigger 1, 1
play 0,0,84 # subtract 16ns: total of else_wait of if/elif conditionals (final else not included)

set_cond 1, 1, 0, 4
wait 16 # total else_wait of if/elif conditionals
play 0,1,68 # subtract 12 ns: total of else of next conditionals
# duration of else: 8 ns

set_cond 1, 2, 0, 4
wait 8 # wait 8 ns: # total else_wait of if/elif conditionals - total else_wait previous conditionals
set_ph 500000000
play 1,0,76 # subtract 4 ns for else of next conditional
# duration of else: 8 ns

set_awg_gain 32000,16000
set_cond 1, 4, 0, 4
play 1,1,80 # starts with 16 ns delay
# duration of else: 4 ns

set_cond 0, 0, 0, 0
play 0,0,100
stop
"""

GAIN_32000 = 32000 / 32768
GAIN_16000 = 16000 / 32768
EXPECTED_IF_ELIF_ELSE = (
    284,
    # cond 1 true: play 0,1 at 104. cond 2 and 4 false: 3 x else_wait 4.
    # set_ph in skipped branch is discarded. Gain set before skipped play is applied by final play 0,0.
    [(4, GAUSS80, GAIN_32000), (104, GAUSS80, GAIN_32000), (184, GAUSS80, GAIN_32000)],
    [(4, GAUSS80, GAIN_32000), (104, ZERO80, GAIN_32000), (184, GAUSS80, GAIN_16000)],
)


def run(cluster, program):
    qrm = cluster.module2
    seq = qrm.sequencers[0]
    seq.sync_en(True)
    seq.mod_en_awg(True)
    seq.connect_out0("I")
    seq.connect_out1("Q")
    seq.thresholded_acq_trigger_en(True)
    seq.thresholded_acq_trigger_address(1)
    seq.trigger1_count_threshold(1)
    seq.sequence({"program": program, "waveforms": WAVEFORMS, "weights": {}, "acquisitions": {}})
    qrm.arm_sequencer(0)

    # Other sequencers participating in sync.
    for seq_number, trigger_number in [(1, 2), (2, 3)]:
        other = qrm.sequencers[seq_number]
        other.sync_en(True)
        other.thresholded_acq_trigger_en(True)
        other.thresholded_acq_trigger_address(trigger_number)
        other.sequence({"program": "wait_sync 100\nstop\n", "waveforms": {}, "weights": {}, "acquisitions": {}})
        qrm.arm_sequencer(seq_number)

    cluster.start_sequencer()
    for seq_nr in range(3):
        assert qrm.get_sequencer_status(seq_nr, 1).err_flags == []
    return seq.get_output()


@pytest.mark.parametrize("program,expected", [
    (PROGRAM_IF_ELSE, EXPECTED_IF_ELSE),
    (PROGRAM_IF_ELIF_ELSE, EXPECTED_IF_ELIF_ELSE),
], ids=["if-else", "if-elif-else"])
def test_conditional_playback(sim_cluster, program, expected):
    duration, plays_path0, plays_path1 = expected
    output = run(sim_cluster({2: "QRM"}), program)

    for path, plays in [("I", plays_path0), ("Q", plays_path1)]:
        expected_data = V_MAX * SCALE * render_path(duration, plays)
        np.testing.assert_allclose(output[path].data, expected_data, rtol=0, atol=3 * lsb(V_MAX),
                                   err_msg=f"path {path}")
