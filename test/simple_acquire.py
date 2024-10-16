from q1simulator import Q1Simulator

import qcodes
qcodes.Instrument.close_all()

q1sim = Q1Simulator("q1sim", sim_type="QRM")

sequence = {
            "waveforms": {},
            "weights": {},
            "acquisitions": {"testacq": {"num_bins": 1, "index": 0}},
            "program": "acquire 0, 0, 1000\nstop",
        }

sqcr = q1sim.sequencer0
sqcr.integration_length_acq = 1000
sqcr.cont_mode_en_awg_path0 = True
sqcr.cont_mode_waveform_idx_awg_path0 = 0
sqcr.cont_mode_en_awg_path1 = True
sqcr.cont_mode_waveform_idx_awg_path1 = 0
sqcr.cont_mode_waveform_idx_awg_path1 = None
sqcr.cont_mode_waveform_idx_awg_path0 = None
sqcr.cont_mode_en_awg_path1 = None
sqcr.cont_mode_en_awg_path0 = None
sqcr.integration_length_acq = None
sqcr.sync_en = None

sqcr.sequence(sequence)
q1sim.arm_sequencer(0)
q1sim.start_sequencer(0)
q1sim.get_sequencer_status(0)
q1sim.get_acquisition_status(0)
data = q1sim.get_acquisitions(0)["testacq"]["acquisition"]["bins"]["integration"]
