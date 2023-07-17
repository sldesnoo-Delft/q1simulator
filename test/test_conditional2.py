import matplotlib.pyplot as pt
import numpy as np
import qcodes as qc
from q1simulator import Cluster as SimCluster
from qblox_instruments import Cluster

def generate_waveforms():
    ''' Generates ramps '''
    waveforms = {}
    for i, (start, stop) in enumerate([
            (-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0),
            (1.0, 0.5), (0.5, 0.0), (0.0, -0.5), (-0.5, -1.0),
            ]):
        waveforms[f'ramp{i}'] = {
                'data': (0.5*np.linspace(start, stop, 60)).tolist(),
                'index': i,
                }
    return waveforms


class TestCluster:

    def __init__(self, sim=True):
        qc.Instrument.close_all()
        self.sim = sim
        if sim:
            cluster = SimCluster('test', {1:'QCM', 2:'QRM'})
            qcm = cluster.module1
            qcm.config('trace', True)
            qrm = cluster.module2
            qrm.config('trace', True)
        else:
            cluster = Cluster('Qblox_Cluster', '192.168.0.3')
            qcm = cluster.module2
            qrm = cluster.module10

        self.cluster = cluster
        self.qcm = qcm
        self.qrm = qrm
        self.reset()

    def reset(self):
        self.cluster.reset()
        self.armed = []
        for module in [self.qcm, self.qrm]:
            seq = module.sequencers[0]
            seq.channel_map_path0_out0_en(True)
            seq = module.sequencers[1]
            seq.channel_map_path1_out1_en(True)
        for num in [0,1]:
            seq = self.qrm.sequencers[num]
            seq.thresholded_acq_trigger_en(False)
            seq.integration_length_acq(100)


    def load(self, on_qcm, sequencer_number, program, waveforms={}, weights={}, acquisitions={}):
        if on_qcm:
            module = self.qcm
        else:
            module = self.qrm
        seq = module.sequencers[sequencer_number]
        seq.sequence({
                'program': program,
                'waveforms': waveforms,
                'weights': weights,
                'acquisitions': acquisitions})
        seq.sync_en(True)
        module.arm_sequencer(sequencer_number)
        self.armed.append([module.slot_idx, sequencer_number])

    def trigger_out(self, sequencer_number, address, threshold:float,
                    invert:bool=False):
        seq = self.qrm.sequencers[sequencer_number]
        seq.thresholded_acq_trigger_en(True)
        seq.thresholded_acq_trigger_address(address)
        seq.thresholded_acq_threshold(threshold)
        seq.thresholded_acq_trigger_invert(invert)

    def set_trigger_thresholding(self, on_qcm, sequencer_number,
                                 address: int, count: int = 1,
                                 invert: bool = False):
        if on_qcm:
            module = self.qcm
        else:
            module = self.qrm
        seq = module.sequencers[sequencer_number]
        seq.set_trigger_thresholding(address, count, invert)

    def run(self):
        self.cluster.start_sequencer()
        if self.sim:
            self.qrm.plot()
            self.qcm.plot()


sim = TestCluster(sim=False)
waveforms = generate_waveforms()
acquisitions = {
        "acq0": {"num_bins": 1, "index": 0},
        }

n_rep = 20_000_000
sim.reset()

#%% IF-ELSE play

qrm0_program=f'''
move {n_rep},R1
wait_sync 100
start:
  acquire 0, 0, 100
  wait 500
loop R1,@start
stop
'''

qcm1_program=f'''
move {n_rep},R1
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
'''
#pt.figure()
print('load',flush=True)
sim.trigger_out(0, 1, 0.9)
sim.set_trigger_thresholding(False, 1, 1)
sim.load(False, 0, qrm0_program, acquisitions=acquisitions)
sim.load(True, 1, qcm1_program, waveforms=waveforms)
print('run',flush=True)
sim.run()
#pt.legend()

stop()

#%% IF-ELSE set_awg_offs

#sim.reset()

qrm0_program=f'''
move {n_rep},R1
wait_sync 100
start:
  acquire 0, 0, 100
  wait 500
loop R1,@start
stop
'''

qcm1_program=f'''
move {n_rep},R1
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
'''
pt.figure()
sim.trigger_out(0, 1, 0.5)
sim.set_trigger_thresholding(False, 1, 1)
sim.load(False, 0, qrm0_program, acquisitions=acquisitions)
sim.load(True, 1, qcm1_program, waveforms=waveforms)
sim.run()
pt.legend()

stop()

#%% Condition evaluation
#sim.reset()

trigger_delay = 248

qrm0_program=f'''
move {n_rep},R1
wait_sync 100
start:
  acquire 0, 0, 100
  wait {trigger_delay+200}
loop R1,@start
stop
'''

qcm1_program=f'''
move {n_rep},R1
wait_sync 100
start:
  set_latch_en 1, 100
  set_awg_gain 32767,32767
  set_mrk 15
  play 1,1,100
  set_mrk 0
  play 3,3,60
  wait {trigger_delay-160}

  # set_cond enable, 0b0001, OR, else_wait
  set_cond 1, 1, 0, 4
  play 1,1,8
  play 1,1,8
  play 1,1,8
  play 1,1,8
  play 1,1,8
  wait 56 # 6 statements, subtract 4 ns for else

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
'''
pt.figure()
sim.trigger_out(0, 1, -0.5)
sim.set_trigger_thresholding(False, 1, 1)
sim.load(False, 0, qrm0_program, acquisitions=acquisitions)
sim.load(True, 1, qcm1_program, waveforms=waveforms)
sim.run()
pt.legend()

stop()
