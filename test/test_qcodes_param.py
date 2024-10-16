from q1simulator import Q1Simulator

sim = Q1Simulator("test", 1, "QCM")

#%%
seq = sim.sequencer0
seq.connect_out0("I")
seq.sync_en(True)
seq.gain_awg_path0(-0.5)
seq.offset_awg_path0(0.1)

sequence = {
    "waveforms": {
        "ones80": {
            "data": [1.0]*80,
            "index": 0
            },
        },
    "weights": {},
    "acquisitions": {},
    "program": """
       wait 20
       play 0, 0, 100
       set_awg_offs 32767, 32767
       upd_param 100
       set_awg_offs 0,0
       upd_param 100
       stop
    """,
    }

seq.sequence(sequence)

sim.arm_sequencer(0)
sim.start_sequencer(0)
sim.get_sequencer_status(0)

sim.plot()
