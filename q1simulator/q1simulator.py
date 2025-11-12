import logging
from functools import partial

import numpy as np
import matplotlib.pyplot as pt
import qcodes as qc

from .channel_data import MarkerOutput, SampledOutput
from .q1sequencer import Q1Sequencer
from .qblox_version import check_qblox_instrument_version
from .triggers import TriggerDistributor
from .trigger_sorting import get_seq_trigger_info, sort_sequencers

from qblox_instruments import (
    InstrumentClass, InstrumentType,
    SystemStatuses, SystemStatus, SystemStatusSlotFlags,
)


logger = logging.getLogger(__name__)


class Q1Module(qc.instrument.InstrumentBase):
    _module_parameters = [
        'ext_trigger_input_delay',
        'ext_trigger_input_trigger_en',
        'ext_trigger_input_trigger_address',
        'marker0_inv_en',
        'marker1_inv_en',
        'marker2_inv_en',
        'marker3_inv_en',
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
        'out0_lo_freq_cal_type_default',
        'out1_lo_freq_cal_type_default',
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
        'in0_offset',
        'in1_offset',
    ]
    _sim_parameters_qrm_rf = [
        'in0_att',
        'out0_att',
        'in0_offset_path0',
        'in0_offset_path1',
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
        'out0_in0_lo_freq_cal_type_default',
    ]

    # NOTE: No __init__() !!!
    # This class is used as a mixin. Although quite heavy mixin.

    def init_module(self, n_sequencers=6, sim_type=None):
        self._sim_type = sim_type
        if sim_type is None:
            raise Exception('sim_type must be specified')

        self._is_qcm = sim_type in ['QCM', 'QCM-RF', 'Viewer']
        self._is_qrm = sim_type in ['QRM', 'QRM-RF', 'Viewer']
        self._is_rf = sim_type in ['QCM-RF', 'QRM-RF']

        if not (self._is_qcm or self._is_qrm):
            raise ValueError(f'Unknown sim_type: {sim_type}')

        sim_params = []
        sim_params += self._module_parameters
        if sim_type == 'QCM':
            sim_params += self._sim_parameters_qcm
        elif sim_type == 'QCM-RF':
            sim_params += self._sim_parameters_qcm_rf
        elif sim_type == 'QRM':
            sim_params += self._sim_parameters_qrm
        elif sim_type == 'QRM-RF':
            sim_params += self._sim_parameters_qrm_rf
        elif sim_type == 'Viewer':
            sim_params += list(set(self._sim_parameters_qcm+self._sim_parameters_qrm))

        n_out_ch_type = {
            'QCM': 4,
            'QRM': 2,
            'QCM-RF': 2,
            'QRM-RF': 1,
            'Viewer': 0,
        }
        for ch in range(n_out_ch_type[sim_type]):
            sim_params += [
                f'out{ch}_latency',
                f'out{ch}_fir_config',
                f'out{ch}_fir_coeffs',
            ]
            sim_params += [f'out{ch}_exp{i}_config' for i in range(4)]
            if sim_type == 'QCM':
                sim_params += [f'out{ch}_exp{i}_time_constant' for i in range(4)]
                sim_params += [f'out{ch}_exp{i}_amplitude' for i in range(4)]

        for ch in range(4):
            sim_params += [
                f'marker{ch}_fir_config',
            ]
            sim_params += [f'marker{ch}_exp{i}_config' for i in range(4)]

        for par_name in sim_params:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        self.sequencers = [Q1Sequencer(self, f'seq{i}', sim_type, i)
                           for i in range(n_sequencers)]
        for i, seq in enumerate(self.sequencers):
            self.add_submodule(f'sequencer{i}', seq)

        if sim_type == 'QCM-RF':
            self.out0_lo_cal = lambda: logger.info("Calibrate LO 0")
            self.out1_lo_cal = lambda: logger.info("Calibrate LO 1")
        if sim_type == 'QRM-RF':
            self.out0_in0_lo_cal = lambda: logger.info("Calibrate LO 0")

        if self._is_qrm:
            if self._is_rf:
                self.in0_att(0)
            else:
                self.in0_gain(0)
                self.in1_gain(0)

        # Extra attributes for testing with simulator
        self._value_for_condition: int | None = None

    @property
    def module_type(self) -> InstrumentType:
        return InstrumentType.QCM if self._is_qcm else InstrumentType.QRM

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
        for seq in self.sequencers:
            seq.reset()

    def _set(self, name, value):
        logger.info(f'{self.name}:{name}={value}')

    def get_num_system_error(self):
        return 0

    def get_system_error(self):
        return '0,"No error"'

    def arm_sequencer(self, sequencer: int | None = None):
        seq_nums = range(len(self.sequencers)) if sequencer is None else (sequencer,)
        for seq_num in seq_nums:
            self.sequencers[seq_num].arm_sequencer()

    def start_sequencer(self, sequencer: int | None = None):
        start_indices = self.armed_sequencers if sequencer is None else (sequencer,)
        for seq_nr in start_indices:
            self.sequencers[seq_nr].start_sequencer()

    def stop_sequencer(self, sequencer: int | None = None):
        seq_nums = range(len(self.sequencers)) if sequencer is None else (sequencer,)
        for seq_num in seq_nums:
            self.sequencers[seq_num].stop_sequencer()

    def get_sequencer_status(self, seq_nr: int, timeout: int = 0, timeout_poll_res: float = 0.02):
        return self.sequencers[seq_nr].get_sequencer_status()

    def get_acquisition_status(self, sequencer: int, timeout: int = 0,
                               timeout_poll_res: float = 0.02, check_seq_state: bool = True):
        return self.sequencers[sequencer].get_acquisition_status()

    def get_waveforms(self, sequencer: int, *, as_numpy: bool = False):
        return self.sequencers[sequencer].get_waveforms(as_numpy=as_numpy)

    def get_weights(self, sequencer: int, *, as_numpy: bool = False):
        return self.sequencers[sequencer].get_weights(as_numpy=as_numpy)

    def get_acquisitions(self, sequencer: int, *, as_numpy: bool = False) -> dict:
        return self.sequencers[sequencer].get_acquisitions(as_numpy=as_numpy)

    def delete_acquisition_data(self, sequencer, name='', all=False):
        self.sequencers[sequencer].delete_acquisition_data(name=name, all=all)

    def store_scope_acquisition(self, sequencer: int, name: str):
        self.sequencers[sequencer].store_scope_acquisition(name)

    def start_adc_calib(self):
        if self._is_qrm:
            logger.info('Calibrate ADC')
        else:
            logger.error("QCM does not have method 'start_adc_calib'")

    def connect_sequencer(self, sequencer: int, *connections: str) -> None:
        self.sequencers[sequencer].connect_sequencer(*connections)

    def disconnect_inputs(self) -> None:
        if not self._is_qrm:
            raise NotImplementedError("Instrument type is not QRM")
        for sequencer in self.sequencers:
            if self._is_rf:
                sequencer.parameters["connect_acq"].set("off")
            else:
                sequencer.parameters["connect_acq_I"].set("off")
                sequencer.parameters["connect_acq_Q"].set("off")

    def disconnect_outputs(self) -> None:
        n_out_ch = 4 if self._is_qcm else 2
        if self._is_rf:
            n_out_ch //= 2
        for sequencer in self.sequencers:
            for i in range(n_out_ch):
                sequencer.parameters[f"connect_out{i}"].set("off")

    # ---- Simulator specific methods ----

    @property
    def armed_sequencers(self):
        return [
            seq_num
            for seq_num, seq in enumerate(self.sequencers)
            if seq.run_state == "ARMED"
        ]

    def config_seq(self, seq_nr, name, value):
        self.sequencers[seq_nr].config(name, value)

    def config(self, name, value):
        for seq in self.sequencers:
            seq.config(name, value)

    @property
    def value_for_condition(self) -> int | None:
        """If not None let conditions evaluate to true for OR, AND, XOR if value is 1
        and for NOR, NAND and XNOR if value is 0.
        """
        return self._value_for_condition

    @value_for_condition.setter
    def value_for_condition(self, value: int | None):
        self._value_for_condition = value
        for seq in self.sequencers:
            seq.set_forced_condition_value(value)

    def get_simulation_end_time(self):
        return max(seq.get_simulation_end_time() for seq in self.sequencers)

    def plot(self,
             t_min: float = None,
             t_max: float = None,
             channels: list[str] | list[int] | None = None,
             analogue_filter: bool = False,
             analogue_output_frequency: float = 4e9,
             output_per_sequencer: bool = True,
             **kwargs):

        output = self.get_output(t_min, t_max, channels,
                                 analogue_filter, analogue_output_frequency,
                                 output_per_sequencer)
        for name, data in output.items():
            if output_per_sequencer:
                is_marker = len(name) > 3 and name[-3:-1] == '-M'
            else:
                is_marker = name[0] == "M"
            linestyle = ":" if is_marker else "-"
            if isinstance(data, MarkerOutput):
                x, y = data.get_xy_lines()
                pt.plot(x, y, linestyle, label=name)
            elif isinstance(data, SampledOutput):
                if data.sample_rate == 1:
                    pt.plot(data.data, linestyle, label=name)
                else:
                    t = data.get_time_data()
                    pt.plot(t, data.data, linestyle, label=name)

        limits = {}
        if t_min is not None:
            limits['left'] = t_min
        if t_max is not None:
            limits['right'] = t_max
        if limits:
            pt.xlim(**limits)

    def get_output(self,
                   t_min: float = None,
                   t_max: float = None,
                   channels: list[str] | list[int] | None = None,
                   analogue_filter: bool = False,
                   output_frequency: float = 4e9,
                   output_per_sequencer: bool = True,
                   ):
        output = {}
        for i, seq in enumerate(self.sequencers):
            if channels is not None and i not in channels and seq.label not in channels:
                # skip channel
                continue
            # assume only sequencers in sync mode have executed.
            if not seq.sync_en():
                continue
            # sequencer may generate output on muliple outputs (I, Q, marker)
            seq_output = seq.get_output(t_min=t_min, t_max=t_max,
                                        analogue_filter=analogue_filter,
                                        output_frequency=output_frequency)
            if output_per_sequencer:
                label = seq.label
                iq_output = seq.get_enabled_outputs() == "IQ"
                for path, data in seq_output.items():
                    if path[0] in "IQ" and not iq_output:
                        output[label] = data
                    else:
                        output[label+"-"+path] = data
            else:
                for i, path in enumerate(seq.output_selected_path):
                    if path == "off":
                        continue
                    out = f"O{i+1}"
                    data = seq_output[path]
                    if out in output:
                        output[out] += data
                    else:
                        output[out] = data
                for i in range(4):
                    marker = f"M{i+1}"
                    if marker not in seq_output:
                        continue
                    data = seq_output[marker]
                    if marker in output:
                        output[marker] += data
                    else:
                        output[marker] = data
        return output

    def print_acquisitions(self):
        for i, seq in enumerate(self.sequencers):
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
                                      threshold=100),
                      ']')
                print("  'path1': [",
                      np.array2string(np.array(bins['integration']['path1']),
                                      prefix=' '*12,
                                      separator=',',
                                      threshold=100),
                      ']')
                print("  'avg_cnt': [",
                      np.array2string(np.array(bins['avg_cnt']),
                                      prefix=' '*12,
                                      separator=',',
                                      threshold=100),
                      ']')

    def print_registers(self, seq_nr, reg_nrs=None):
        self.sequencers[seq_nr].print_registers(reg_nrs)


class Q1Simulator(qc.Instrument, Q1Module):
    _pulsar_parameters = [
        'reference_source',
    ]
    _log_only_params = [
        'led_brightness',
    ]

    def __init__(self, name, n_sequencers=6, sim_type=None):
        check_qblox_instrument_version()
        super().__init__(name)
        super().init_module(n_sequencers, sim_type)

        for par_name in self._pulsar_parameters:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        for par_name in self._log_only_params:
            self.add_parameter(par_name,
                               set_cmd=partial(self._log_set, par_name))
        self.ignore_triggers = False

    def get_idn(self):
        return dict(vendor='Q1Simulator', model=self._sim_type, serial='', firmware='')

    @property
    def instrument_class(self):
        return InstrumentClass.PULSAR

    @property
    def instrument_type(self):
        return InstrumentType[self._sim_type]

    def reset(self):
        self.invalidate_cache()
        super().reset()

    def get_system_status(self):
        return SystemStatus(
            SystemStatuses.OKAY,
            [],
            SystemStatusSlotFlags({}))

    def start_sequencer(self, sequencer: int | None = None):
        if sequencer is not None:
            self.sequencers[sequencer].run()
            return

        # Get list of armed sequencers
        # pass to sequence executor

        sequencers = [self.sequencers[seq_number] for seq_number in self.armed_sequencers]
        run_sequencers(sequencers, self.ignore_triggers)

    def _log_set(self, name, value):
        logger.info(f'{self.name}: {name}={value}')

    def plot(self,
             t_min: float = None,
             t_max: float = None,
             channels: list[str] | list[int] | None = None,
             analogue_filter: bool = False,
             analogue_output_frequency: float = 4e9,
             output_per_sequencer: bool = True,
             **kwargs):
        """Plots the simulated output of the module.

        Args:
            t_min: minimum time in the plot.
            t_max: maximum time in the plot.
            channels: If not None specifies the channels to plot by name or sequencer number.
            analogue_filter: plot result after applying (estimated) analog filter.
            analogue_output_frequency: sample rate of analogue output
            output_per_sequencer:
                if True: plot data for individual sequencers.
                if False: plot data for physical front panel output.
        """
        pt.figure()
        super().plot(t_min=t_min, t_max=t_max, channels=channels,
                     analogue_filter=analogue_filter,
                     analogue_output_frequency=analogue_output_frequency,
                     output_per_sequencer=output_per_sequencer)
        pt.grid(True)
        pt.legend()
        pt.xlabel('[ns]')
        pt.ylabel('[V]')
        pt.show()


def run_sequencers(sequencers: list[Q1Sequencer], ignore_triggers=False):
    if ignore_triggers:
        for seq in sequencers:
            seq.run()
        return

    # sort on used triggers

    # TODO Refactor trigger distribution in runtime distribution.
    seq_infos = [get_seq_trigger_info(seq) for seq in sequencers]
    trigger_dist = TriggerDistributor()
    seq_infos = sort_sequencers(seq_infos)
    for seq_info in seq_infos:
        seq = seq_info.sequencer
        seq.set_trigger_events(trigger_dist.get_trigger_events())
        seq.run()
        trigger_dist.add_emitted_triggers(seq.get_acq_trigger_events())
