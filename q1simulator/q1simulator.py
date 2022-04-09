import logging
from functools import partial

import numpy as np
import qcodes as qc

from .q1sequencer import Q1Sequencer

try:
    from qblox_instruments import (
            SystemStatus, SystemState, SystemStatusSlotFlags,
            InstrumentClass, InstrumentType,
            )

    _legacy_code = False
except:
    print('Q1Simulator assumes qblox_instruments < v0.6')
    _legacy_code = True


class Q1Simulator(qc.Instrument):
    _sim_parameters_both = [
        'reference_source',
        'out0_offset',
        'out1_offset',
        ]
    _sim_parameters_qcm = [
        'out2_offset',
        'out3_offset',
        ]
    _sim_parameters_qrm = [
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

    def __init__(self, name, n_sequencers=6, sim_type=None):
        super().__init__(name)
        self._sim_type = sim_type
        if sim_type is None and not _legacy_code:
            raise Exception('sim_type must be specified')

        self._is_qcm = sim_type in [None, 'QCM', 'Viewer']
        self._is_qrm = sim_type in [None, 'QRM', 'Viewer']

        sim_params = self._sim_parameters_both.copy()
        if self._is_qcm:
            sim_params += self._sim_parameters_qcm
        if self._is_qrm:
            sim_params += self._sim_parameters_qrm

        for par_name in sim_params:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        self.sequencers = [Q1Sequencer(self, f'seq{i}', sim_type)
                           for i in range(n_sequencers)]
        if _legacy_code:
            for i,seq in enumerate(self.sequencers):
                for par_name in seq._seq_parameters:
                    name = f'sequencer{i}_{par_name}'
                    self.add_parameter(name, set_cmd=partial(self._seq_set, name))
        else:
            for i,seq in enumerate(self.sequencers):
                self.add_submodule(f'sequencer{i}', seq)

        self.armed_seq = set()
        if self._is_qrm:
            self.in0_gain(0)
            self.in1_gain(0)

    @property
    def instrument_class(self):
        return InstrumentClass.PULSAR

    @property
    def instrument_type(self):
        return InstrumentType[self._sim_type]

    @property
    def is_qcm_type(self):
        return self._is_qcm

    @property
    def is_qrm_type(self):
        return self._is_qrm

    @property
    def is_rf_type(self):
        return False

    def reset(self):
        self.armed_seq = set()
        for seq in self.sequencers:
            seq.reset()

    def _set(self, name, value):
        logging.info(f'{self.name}:{name}={value}')

    def _seq_set(self, name, value):
        seq_nr = int(name[9])
        self.sequencers[seq_nr]._set_legacy(name[11:], value)

    def get_system_status(self):
        ''' Legacy method qblox_instruments version < v0.6 '''
        return {'status':'OKAY', 'flags':[]}

    def get_system_state(self):
        return SystemState(
            SystemStatus.OKAY,
            [],
            SystemStatusSlotFlags({}))

    def arm_sequencer(self, seq_nr):
        self.armed_seq.add(seq_nr)
        self.sequencers[seq_nr].arm()

    def start_sequencer(self):
        for seq_nr in self.armed_seq:
            self.sequencers[seq_nr].run()

    def stop_sequencer(self):
        self.armed_seq = set()

    def get_sequencer_state(self, seq_nr, timeout=0):
        if not _legacy_code:
            return self.sequencers[seq_nr].get_state()
        else:
            return self.sequencers[seq_nr].get_state_legacy()

    def get_acquisition_state(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_acquisition_state()

    def get_acquisitions(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_acquisition_data()

    def config_seq(self, seq_nr, name, value):
        self.sequencers[seq_nr].config(name, value)

    def config(self, name, value):
        for seq in self.sequencers:
            seq.config(name, value)

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
