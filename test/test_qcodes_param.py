"""
QCoDeS parameters gain_awg_path0 and offset_awg_path0 combined with Q1ASM set_awg_offs.

Simulator only: checks sequencer.get_output().
"""
import numpy as np

from q1_helpers import lsb


V_MAX = 2.5  # QCM
GAIN = -0.5
OFFSET = 0.1

PROGRAM = """
       wait 20
       play 0, 0, 100
       set_awg_offs 32767, 32767
       upd_param 100
       set_awg_offs 0,0
       upd_param 100
       stop
"""


def test_qcodes_gain_and_offset(sim_cluster):
    cluster = sim_cluster({2: "QCM"})
    # Program has no wait_sync: render from start of sequence.
    cluster.config("skip_wait_sync", False)
    seq = cluster.module2.sequencer0
    seq.connect_out0("I")
    seq.sync_en(True)
    seq.gain_awg_path0(GAIN)
    seq.offset_awg_path0(OFFSET)
    seq.sequence({
        "waveforms": {"ones80": {"data": [1.0] * 80, "index": 0}},
        "weights": {},
        "acquisitions": {},
        "program": PROGRAM,
    })
    cluster.arm_sequencer(2, 0)
    cluster.start_sequencer(2, 0)
    assert seq.get_sequencer_status(timeout=1).err_flags == []

    expected = np.empty(320)
    expected[0:20] = OFFSET                  # wait 20: qcodes offset only
    expected[20:100] = OFFSET + GAIN * 1.0   # 80 samples of 1.0 with qcodes gain
    expected[100:120] = OFFSET               # waveform ended
    expected[120:220] = 1.0                  # set_awg_offs 32767 + qcodes offset: clipped
    expected[220:320] = OFFSET               # set_awg_offs 0
    np.testing.assert_allclose(seq.get_output()["I"].data, V_MAX * expected, rtol=0, atol=2 * lsb(V_MAX))
