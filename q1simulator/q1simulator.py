import logging
from functools import partial
from typing import Optional

import numpy as np
import qcodes as qc

from .q1sequencer import Q1Sequencer
from .qblox_version import check_qblox_instrument_version
from .triggers import TriggerDistributor
from .trigger_sorting import get_seq_trigger_info, sort_sequencers

from qblox_instruments import (
        SystemStatus, SystemState, SystemStatusSlotFlags,
        InstrumentClass, InstrumentType,
        )

logger = logging.getLogger(__name__)

class Q1Module(qc.instrument.InstrumentBase):
    _module_parameters = [
        'ext_trigger_input_delay',
        'ext_trigger_input_trigger_en',
        'ext_trigger_input_trigger_address',
        ]
    _sim_parameters_qcm = [
        'out0_offset',
        'out1_offset',
        'out2_offset',
        'out3_offset',
        ]
    _sim_parameters_qcm_rf = [
        'out0_lo_freq',
        'out1_lo_freq',
        'out0_lo_en',
        'out1_lo_en',
        'out0_att',
        'out1_att',
        'out0_offset_path0',
        'out0_offset_path1',
        'out1_offset_path0',
        'out1_offset_path1',
        ]
    _sim_parameters_qrm = [
        'out0_offset',
        'out1_offset',
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
    _sim_parameters_qrm_rf = [
        'in0_att',
        'out0_att',
        'out0_in0_lo_freq',
        'out0_in0_lo_en',
        'out0_offset_path0',
        'out0_offset_path1',
        'scope_acq_trigger_mode_path0',
        'scope_acq_trigger_mode_path1',
        'scope_acq_trigger_level_path0',
        'scope_acq_trigger_level_path1',
        'scope_acq_sequencer_select',
        'scope_acq_avg_mode_en_path0',
        'scope_acq_avg_mode_en_path1',
        ]

    # NOTE: No __init__() !!!
    # This class is used as a mixin. Although quite heavy mixin.

    def init_module(self, n_sequencers=6, sim_type=None):
        check_qblox_instrument_version()
        self._sim_type = sim_type
        if sim_type is None:
            raise Exception('sim_type must be specified')

        self._is_qcm = sim_type in ['QCM', 'QCM-RF', 'Viewer']
        self._is_qrm = sim_type in ['QRM', 'QRM-RF', 'Viewer']
        self._is_rf = sim_type in ['QCM-RF', 'QRM-RF']

        if not (self._is_qcm or self._is_qrm):
            raise ValueError(f'Unknown sim_type: {sim_type}')

        if sim_type == 'QCM':
            sim_params = self._sim_parameters_qcm
        elif sim_type == 'QCM-RF':
            sim_params = self._sim_parameters_qcm_rf
        elif sim_type == 'QRM':
            sim_params = self._sim_parameters_qrm
        elif sim_type == 'QRM-RF':
            sim_params = self._sim_parameters_qrm_rf
        elif sim_type == 'Viewer':
            sim_params = list(set(self._sim_parameters_qcm+self._sim_parameters_qrm))
        else:
            sim_params = []

        for par_name in sim_params:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        self.sequencers = [Q1Sequencer(self, f'seq{i}', sim_type)
                           for i in range(n_sequencers)]
        for i,seq in enumerate(self.sequencers):
            self.add_submodule(f'sequencer{i}', seq)

        self.armed_seq = set()
        if self._is_qrm:
            self.in0_gain(0)
            self.in1_gain(0)

    @property
    def module_type(self) -> InstrumentType:
        return InstrumentType[self._sim_type]

    @property
    def is_qcm_type(self):
        return self._is_qcm

    @property
    def is_qrm_type(self):
        return self._is_qrm

    @property
    def is_rf_type(self):
        return self._is_rf

    def reset(self):
        self.armed_seq = set()
        for seq in self.sequencers:
            seq.reset()

    def _set(self, name, value):
        logger.info(f'{self.name}:{name}={value}')

    def _seq_set(self, name, value):
        seq_nr = int(name[9])
        self.sequencers[seq_nr]._set_legacy(name[11:], value)

    def get_num_system_error(self):
        return 0

    def get_system_error(self):
        return '0,"No error"'

    def get_system_state(self):
        return SystemState(
            SystemStatus.OKAY,
            [],
            SystemStatusSlotFlags({}))

    def arm_sequencer(self, seq_nr):
        self.armed_seq.add(seq_nr)
        self.sequencers[seq_nr].arm()

    def start_sequencer(self, sequencer: Optional[int] = None):
        start_indices = self.armed_seq if sequencer is None else (sequencer,)
        for seq_nr in start_indices:
            self.sequencers[seq_nr].run()

    def stop_sequencer(self, sequencer: Optional[int] = None):
        self.armed_seq = set()

    def get_sequencer_state(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_state()

    def get_acquisition_state(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_acquisition_state()

    def get_acquisitions(self, seq_nr, timeout=0):
        return self.sequencers[seq_nr].get_acquisition_data()

    def delete_acquisition_data(self, seq_nr, name='', all=False):
        self.sequencers[seq_nr].delete_acquisition_data(name=name, all=all)

    def start_adc_calib(self):
        if self._is_qrm:
            logger.info('Calibrate ADC')
        else:
            logger.error("QCM does not have method 'start_adc_calib'")

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


class Q1Simulator(qc.Instrument, Q1Module):
    _pulsar_parameters = [
        'reference_source',
        ]
        
    def __init__(self, name, n_sequencers=6, sim_type=None):
        super().__init__(name)
        super().init_module(n_sequencers, sim_type)

        for par_name in self._pulsar_parameters:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

    def get_idn(self):
        return dict(vendor='Q1Simulator', model=self._sim_type, serial='', firmware='')

    @property
    def instrument_class(self):
        return InstrumentClass.PULSAR

    @property
    def instrument_type(self):
        return InstrumentType[self._sim_type]

    def start_sequencer(self, sequencer: Optional[int] = None):
        if sequencer is not None:
            self.sequencers[sequencer].run()
            return

        # collect sequencers and sort on used triggers
        # TODO Refactor trigger distribution in runtime distribution.
        seq_infos = []
        for seq_number in self.armed_seq:
            seq_infos.append(
                    get_seq_trigger_info(self, seq_number, self.sequencers[seq_number])
                    )

        trigger_dist = TriggerDistributor()
        seq_infos = sort_sequencers(seq_infos)
        for seq_info in seq_infos:
            seq = seq_info.module.sequencers[seq_info.seq_number]
            seq.set_trigger_events(trigger_dist.get_trigger_events())
            seq.run()
            trigger_dist.add_emitted_triggers(seq.get_acq_trigger_events())
