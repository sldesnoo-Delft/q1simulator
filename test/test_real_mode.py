"""
Real mode: path 0 modulated by the NCO plus unmodulated path 1 on output 0.

Simulator only: checks sequencer.get_output().
"""
import numpy as np

from q1_helpers import lsb


V_MAX = 2.5  # QCM
NCO_FREQUENCY = 40e6
ENVELOPE_FREQUENCY = 1e6  # frequency of the sine waveform used for both path_0 and path_1
AMPLITUDE = 0.5
AMPLITUDE_RATIO = 0.5
N_POINTS = int(1e9 / ENVELOPE_FREQUENCY)  # 1 GSa/s
T_WAIT_SYNC = 4  # ns before output starts. The NCO runs from start of sequence.

GAIN0 = int(AMPLITUDE_RATIO * AMPLITUDE * 32767)
GAIN1 = int(AMPLITUDE * 32767)

PROGRAM = f"""
wait_sync     {T_WAIT_SYNC}
set_awg_gain  {GAIN0},{GAIN1}
play:
    set_mrk   15
    play      0,0,4
    set_mrk   0
    upd_param 4
    wait      {N_POINTS - 8}
    stop
"""


def test_real_mode(sim_cluster):
    cluster = sim_cluster({2: "QCM"})
    module = cluster.module2
    seq = module.sequencer0

    envelope = np.sin(2 * np.pi * ENVELOPE_FREQUENCY * np.arange(N_POINTS) * 1e-9)
    seq.sync_en(True)
    seq.mod_en_awg(True)
    seq.nco_freq(NCO_FREQUENCY)
    module.disconnect_outputs()
    seq.real_mode_en(True)
    seq.sequence({
        "waveforms": {"wf0": {"data": envelope.tolist(), "index": 0}},
        "weights": {},
        "acquisitions": {},
        "program": PROGRAM,
    })
    seq.connect_out0("I")
    module.arm_sequencer(0)
    module.start_sequencer(0)
    assert module.get_sequencer_status(0, 1).err_flags == []

    t = np.arange(N_POINTS) + T_WAIT_SYNC
    nco = np.cos(2 * np.pi * NCO_FREQUENCY * t * 1e-9)
    # Modulation scales both paths by 1/sqrt(2).
    expected = V_MAX / np.sqrt(2) * (GAIN0 / 32768 * envelope * nco + GAIN1 / 32768 * envelope)

    output = seq.get_output()
    assert set(output) == {"I", "M1", "M2", "M3", "M4"}
    # Tolerance: rounding of integer arithmetic in rendering.
    np.testing.assert_allclose(output["I"].data, expected, rtol=0, atol=8 * lsb(V_MAX))
