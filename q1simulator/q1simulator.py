import logging
from functools import partial

import numpy as np
import qcodes as qc

from .q1sequencer import Q1Sequencer


class Q1Simulator(qc.Instrument):
    _sim_parameters = [
        'reference_source',
        'out0_offset',
        'out1_offset',
        # -- QCM
        'out2_offset',
        'out3_offset',
        # -- QRM
        'in0_gain',
        'in1_gain',
        'scope_acq_trigger_mode_path0',
        'scope_acq_trigger_mode_path1',
        'scope_acq_trigger_level_path0',
        'scope_acq_trigger_level_path1',
        'scope_acq_sequencer_select',
        'scope_acq_avg_mode_en_path0',
        'scope_acq_avg_mode_en_path1',
        ]

    def __init__(self, name, n_sequencers=6):
        super().__init__(name)
        self.sequencers = [Q1Sequencer(f'{name}-{i}') for i in range(n_sequencers)]
        self.armed_seq = set()

    def reset(self):
        self.armed_seq = set()
        for seq in self.sequencers:
            seq.reset()

    def __getattr__(self, name):
        if name in self._sim_parameters:
            return partial(self.set, name)
        if name.startswith('sequencer'):
            seq_nr = int(name[9])
            return getattr(self.sequencers[seq_nr], name[11:])
        raise AttributeError()

    def set(self, name, value):
        if name.startswith('sequencer'):
            seq_nr = int(name[9])
            self.sequencers[seq_nr].set(name[11:], value)
        else:
            logging.info(f'{self.name}:{name}={value}')

    def get_system_status(self):
        return 'OK'

    def arm_sequencer(self, seq_nr):
        self.armed_seq.add(seq_nr)
        self.sequencers[seq_nr].arm()

    def start_sequencer(self):
        for seq_nr in self.armed_seq:
            self.sequencers[seq_nr].run()

    def stop_sequencer(self):
        self.armed_seq = set()

    def get_sequencer_state(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_state()

    def get_acquisition_state(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_acquisition_state()

    def get_acquisitions(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_acquisition_data()

    def plot(self, **kwargs):
        for seq in self.sequencers:
            seq.plot()

    def print_acquisitions(self):
        for i,seq in enumerate(self.sequencers):
            data = self.get_acquisitions(i)
            if not len(data):
                continue
            for name, datadict in data.items():
                print(f"Acquisitions '{seq.name}':'{name}'")
                bins = datadict['acquisition']['bins']

                print("  'path0': [",
                      np.array2string(np.array(bins['integration']['path0']),
                                        prefix=' '*12,
                                        separator=',',
                                        threshold=100),']')
                print("  'path1': [",
                      np.array2string(np.array(bins['integration']['path1']),
                                      prefix=' '*12,
                                        separator=',',
                                        threshold=100),']')
                print("  'avg_cnt': [",
                      np.array2string(np.array(bins['avg_cnt']),
                                        prefix=' '*12,
                                        separator=',',
                                        threshold=100),']')

    def print_registers(self, seq_nr, reg_nrs=None):
        self.sequencers[seq_nr].print_registers(reg_nrs)
