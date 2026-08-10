import numpy as np
import matplotlib.pyplot as plt
from q1simulator import Cluster

cluster = Cluster("Qblox", modules={2: "QCM"})
module = cluster.module2

nco_frequency = 40e6  # NCO frequency of 40 MHz
sine_envelope_frequency = 1e6  # frequency of the sine waveform used for both path_0 and path_1
amplitude = 0.5  # amplitude of the sine wave on path_0 (modulated by the NCO)

amplitude_ratio = 0.5  # ratio of the amplitude of the sine wave on path_1 (unmodulated)
# with respect to the amplitude of the sine wave on path_0

number_of_pts = int(1e9 / sine_envelope_frequency)  # the AWG sampling rate is 1 Gs/second



wf0 = np.sin(2 * np.pi * sine_envelope_frequency * np.arange(number_of_pts) * 1e-9).tolist()

wfs = {
    "wf0": {"data": wf0, "index": 0},
}

program0 = f"""
wait_sync     4
set_awg_gain  {int(amplitude_ratio * amplitude * 32767)},{int(amplitude * 32767)}
play:
    set_mrk   15
    play      0,0,4
    set_mrk   0
    upd_param 4
    wait      {number_of_pts - 8}
#jmp           @play                # loop indefinitely
    stop
"""

# assembling the program and the waveforms into a sequence dictionary object, as expected by the `qblox-instruments` API
sequence0 = {"waveforms": wfs, "weights": {}, "acquisitions": {}, "program": program0}

module.sequencer0.sync_en(True)
module.sequencer0.mod_en_awg(True)
module.sequencer0.nco_freq(nco_frequency)

module.disconnect_outputs()
module.sequencer0.real_mode_en(True)
module.sequencer0.sequence(sequence0)
module.sequencer0.connect_out0("I")

module.arm_sequencer(0)
module.start_sequencer(0)

module.stop_sequencer(0)
module.sequencer0.real_mode_en(False)

module.plot()
plt.grid(True)
