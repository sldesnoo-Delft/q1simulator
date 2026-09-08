import matplotlib.pyplot as pt
import qcodes as qc
from q1simulator import Cluster as SimCluster
from qblox_instruments import Cluster

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='test_log.txt', encoding='utf-8', level=logging.INFO)


class TestCluster:

    def __init__(self, sim=True):
        qc.Instrument.close_all()
        self.sim = sim
        if sim:
            cluster = SimCluster('test', {2: 'QCM', 6: 'QRM'})
            qcm = cluster.module2
            qcm.config('trace', True)
            qrm = cluster.module6
            qrm.config('trace', True)
        else:
            cluster = Cluster('Qblox_Cluster', '192.168.0.2')
            qcm = cluster.module2
            qrm = cluster.module6

        self.cluster = cluster
        self.qcm = qcm
        self.qrm = qrm
        self.reset()

    def set_trace(self, enabled: bool):
        if self.sim:
            self.qcm.config("trace", enabled)
            self.qrm.config("trace", enabled)

    def reset(self):
        self.cluster.reset()
        self.armed = []
        for module in [self.qcm, self.qrm]:
            seq = module.sequencers[0]
            seq.connect_out0("I")
            seq = module.sequencers[1]
            seq.connect_out1("Q")
        for num in [0, 1]:
            seq = self.qrm.sequencers[num]
            seq.thresholded_acq_trigger_en(False)
            seq.integration_length_acq(100)

    def load(self, module, sequencer_number, program, waveforms={}, weights={}, acquisitions={}):
        seq = module.sequencers[sequencer_number]
        seq.sequence({
                'program': program,
                'waveforms': waveforms,
                'weights': weights,
                'acquisitions': acquisitions})
        seq.sync_en(True)
        module.arm_sequencer(sequencer_number)
        self.armed.append([module.slot_idx, sequencer_number])

    def trigger_out(self, sequencer_number, address, threshold: float,
                    invert: bool = False):
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

    def routes(self, routes: dict[int, list]):
        self.cluster.clear_router()
        for event_id, targets in routes.items():
            self.cluster.set_cmm_route(event_id, targets)

    def run(self):
        self.cluster.start_sequencer()
        for slot_idx, seq_num in self.armed:
            self.cluster.get_sequencer_status(slot_idx, seq_num, 1)
        if self.sim:
            self.qrm.plot()
            self.qcm.plot()
        pt.legend()


sim = TestCluster(sim=True)
waveforms = {}
acquisitions = {
        "acq0": {"num_bins": 1, "index": 0},
        }

n_rep = 1
head = f"""
move {n_rep},R1
wait_sync 100
"""

tail = """
loop R1,@start
stop
"""

# %% Basic send/receive

# 20 -> qcm0, qcm1
# 21 -> qcm1
# 22 -> qrm0


qrm0_program = head + """
start:
  acquire 0,0,100
  fb_com_data 20,40,400
  fb_com_data 21,41,400
  fb_com_data 22,42,400
  wait 800

  fb_pull_data R12,R13 # receive #22
""" + tail

qcm0_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pull_data R10,R11 # receive #20

""" + tail

qcm1_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pop_data 20,R11 # receive #20
  fb_pull_data R12,R13 # receive #21
""" + tail

sim.reset()

print('load', flush=True)
sim.load(sim.qrm, 0, qrm0_program, acquisitions=acquisitions)
sim.load(sim.qcm, 0, qcm0_program)
sim.load(sim.qcm, 1, qcm1_program)
sim.routes({
    20: [sim.qcm.sequencer0, sim.qcm.sequencer1],
    21: [sim.qcm.sequencer1],
    22: [sim.qrm.sequencer0],
    })

print('run', flush=True)
sim.run()

regs_qcm0 = sim.qcm.get_sequencer_registers(0, ["R10", "R11", "R12", "R13"])
regs_qcm1 = sim.qcm.get_sequencer_registers(1, ["R10", "R11", "R12", "R13"])
regs_qrm0 = sim.qrm.get_sequencer_registers(0, ["R10", "R11", "R12", "R13"])
print(regs_qcm0)
print(regs_qcm1)
print(regs_qrm0)
assert regs_qcm0["R10"] == 20
assert regs_qcm0["R11"] == 40
assert regs_qcm1["R11"] == 40
assert regs_qcm1["R12"] == 21
assert regs_qcm1["R13"] == 41
assert regs_qrm0["R12"] == 22
assert regs_qrm0["R13"] == 42

# %% Two at fb events at same time: What happens on HW?

# TODO Also test with larger shift: 32 bits. Check OR.

# 20 -> qcm1
# 21 -> qcm1


qrm0_program = head + """
start:
  acquire 0,0,100
  fb_com_data 20,40,400
  wait 800

""" + tail

qcm0_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 100
  fb_com_data 21,41,400

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pull_data R10,R11 # receive #20

""" + tail

qcm1_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

  fb_pull_data R10,R11 # receive #20
  fb_pull_data R12,R13 # receive #21
""" + tail

sim.reset()

print('load', flush=True)
sim.load(sim.qrm, 0, qrm0_program, acquisitions=acquisitions)
sim.load(sim.qcm, 0, qcm0_program)
sim.load(sim.qcm, 1, qcm1_program)
sim.routes({
    20: [sim.qcm.sequencer1],
    21: [sim.qcm.sequencer1],
    })

print('run', flush=True)
sim.run()

regs_qcm0 = sim.qcm.get_sequencer_registers(0, ["R10", "R11", "R12", "R13"])
regs_qcm1 = sim.qcm.get_sequencer_registers(1, ["R10", "R11", "R12", "R13"])
regs_qrm0 = sim.qrm.get_sequencer_registers(0, ["R10", "R11", "R12", "R13"])
print(regs_qcm0)
print(regs_qcm1)
print(regs_qrm0)
assert regs_qcm1["R10"] == 21
assert regs_qcm1["R11"] == 41
assert regs_qcm1["R12"] == 20
assert regs_qcm1["R13"] == 40

# %% 2 events using Write Combine

# 20 -> qcm1
# 21 -> qcm1


qrm0_program = head + """
start:
  acquire 0,0,96
  fb_com_cfg 1, 0, 2, 4
  fb_com_data 20,5,400
  wait 2000

""" + tail

qcm0_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 96
  fb_com_cfg 1, 8, 2, 4
  fb_com_data 20,10,400

  set_awg_offs 0,0
  set_mrk 0
  upd_param 2000

""" + tail

qcm1_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 1000

  fb_pull_data R10,R11 # receive #20
""" + tail

sim.reset()

print('load', flush=True)
sim.load(sim.qrm, 0, qrm0_program, acquisitions=acquisitions)
sim.load(sim.qcm, 0, qcm0_program)
sim.load(sim.qcm, 1, qcm1_program)
sim.routes({
    20: [sim.qcm.sequencer1],
    21: [sim.qcm.sequencer1],
    })

print('run', flush=True)
sim.run()

regs_qcm0 = sim.qcm.get_sequencer_registers(0, ["R10", "R11", "R12", "R13"])
regs_qcm1 = sim.qcm.get_sequencer_registers(1, ["R10", "R11", "R12", "R13"])
regs_qrm0 = sim.qrm.get_sequencer_registers(0, ["R10", "R11", "R12", "R13"])
print(regs_qcm0)
print(regs_qcm1)
print(regs_qrm0)
assert regs_qcm1["R10"] == 20
assert regs_qcm1["R11"] == 2565
assert regs_qcm1["R12"] == 0
assert regs_qcm1["R13"] == 0

# %% Fast write repetition of N events

n_events = 10
t_wait = 100
# 20 -> qcm1


qrm0_program = head + """
start:
  acquire 0,0,100
"""
for i in range(n_events):
    qrm0_program += f"""
  fb_com_data 20,{100+i},{t_wait}
"""
qrm0_program += f"""
  wait {20000-n_events*t_wait}
""" + tail

qcm0_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 1
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 20000

""" + tail

qcm1_program = head + """
start:
  set_awg_offs 32767,32767
  set_mrk 15
  upd_param 100

  set_awg_offs 0,0
  set_mrk 0
  upd_param 1000
"""

for i in range(n_events):
    qcm1_program += f"""
  fb_pop_data 20,R{10+i}
  wait {t_wait}
"""
qcm1_program += f"""
  wait {20000-1000-n_events*t_wait}
""" + tail

sim.reset()
sim.set_trace(False)
print('load', flush=True)
sim.load(sim.qrm, 0, qrm0_program, acquisitions=acquisitions)
sim.load(sim.qcm, 0, qcm0_program)
sim.load(sim.qcm, 1, qcm1_program)
sim.routes({
    20: [sim.qcm.sequencer1],
    21: [sim.qcm.sequencer1],
    })

print('run', flush=True)
sim.run()

regs_qcm1 = sim.qcm.get_sequencer_registers(1, [f"R{10+i}" for i in range(n_events)])
print(regs_qcm1)
