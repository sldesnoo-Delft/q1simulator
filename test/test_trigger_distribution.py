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
            # qcm.config('trace', True)
            qrm = cluster.module2
            # qrm.config('trace', True)
            cluster.config('render_repetitions', False)
            cluster.config('skip_wait_sync', True)
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
            seq.connect_out0('I')
            seq.label = f"{module.name}-0"
            seq = module.sequencers[1]
            seq.connect_out1('Q')
            seq.label = f"{module.name}-1"
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
            pt.figure()
            self.qcm.plot()
            self.qrm.plot()
            pt.legend()
            pt.grid(True)

def qcm_program(used_triggers):
    mask = 0
    for addr in used_triggers:
        mask |= 1 << (addr - 1)

    return f'''
    move {n_rep},R1
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

      # set_cond enable, 0b0001, NOR, else_wait
      set_cond 1, {mask}, 1, 4
      play 6,6,80
      wait 20

      # disable conditional
      set_cond 0, 0, 0, 0
      play 3,3,96
      latch_rst 4
    loop R1,@_start
    stop
    '''

def qrm_program(used_triggers):
    mask = 0
    for addr in used_triggers:
        mask |= 1 << (addr - 1)

    return f'''
    move {n_rep},R1
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

      # set_cond enable, 0b0001, NOR, else_wait
      set_cond 1, {mask}, 1, 4
      play 6,6,80
      wait 20

      # disable conditional
      set_cond 0, 0, 0, 0
      play 3,3,96
      latch_rst 4
    loop R1,@_start
    stop
    '''

sim = TestCluster(sim=True)
waveforms = generate_waveforms()
acquisitions = {
        "acq0": {"num_bins": 1, "index": 0},
        }

n_rep = 20 #_000_000


# %%

# qcm0: uses [1,2], qrm0: emits 1, qrm1 emits 2,

sim.reset()
sim.load(True, 0, qcm_program([1, 2]), waveforms=waveforms)
sim.load(False, 0, qrm_program([]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(0, 1, 0.0)
sim.load(False, 1, qrm_program([]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(1, 2, 0.0)
sim.cluster.config('acq_trigger_value', 0)
sim.run()
sim.cluster.config('acq_trigger_value', 1)
sim.run()

# %%

# qcm0: uses 1, qrm1: emits 1 uses 2, qrm2 emits 2

sim.reset()
sim.load(True, 0, qcm_program([1]), waveforms=waveforms)
sim.load(False, 0, qrm_program([2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(0, 1, 0.0)
sim.load(False, 1, qrm_program([]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(1, 2, 0.0)
sim.cluster.config('acq_trigger_value', 0)
sim.run()
sim.cluster.config('acq_trigger_value', 1)
sim.run()

# %%

# qcm0: uses 1, qrm1: emits 1 uses 2, qrm2 emits 2, uses 2

sim.reset()
sim.load(True, 0, qcm_program([1]), waveforms=waveforms)
sim.load(False, 0, qrm_program([2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(0, 1, 0.0)
sim.load(False, 1, qrm_program([2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(1, 2, 0.0)
sim.cluster.config('acq_trigger_value', 0)
sim.run()
sim.cluster.config('acq_trigger_value', 1)
sim.run()

# %%

# qcm0: uses 2 qrm1: emits 1 uses 1, 2, qrm2 emits 2, uses 2

sim.reset()
sim.load(True, 0, qcm_program([2]), waveforms=waveforms)
sim.load(False, 0, qrm_program([1, 2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(0, 1, 0.0)
sim.load(False, 1, qrm_program([2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(1, 2, 0.0)
sim.cluster.config('acq_trigger_value', 0)
sim.run()
sim.cluster.config('acq_trigger_value', 1)
sim.run()

# %%

# Dependencies on not configured triggers

sim.reset()
sim.load(True, 0, qcm_program([1, 2, 3]), waveforms=waveforms)
sim.load(False, 0, qrm_program([1, 2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(0, 1, 0.0)
sim.load(False, 1, qrm_program([2, 9]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(1, 2, 0.0)
sim.cluster.config('acq_trigger_value', 0)
sim.run()
sim.cluster.config('acq_trigger_value', 1)
sim.run()


# %%

# qcm0: uses 1,2 qrm1: uses 1, 2, qrm2 uses 2

# !!! SORTING FAILS !!!
sim.reset()
sim.load(True, 0, qcm_program([]), waveforms=waveforms)
sim.load(False, 0, qrm_program([2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(0, 1, 0.0)
sim.load(False, 1, qrm_program([1, 2]), acquisitions=acquisitions, waveforms=waveforms)
sim.trigger_out(1, 2, 0.0)
sim.run()


