import json
import logging
from dataclasses import dataclass
from functools import partial
from typing import Iterator, Iterable

import numpy as np

from qcodes.instrument.channel import InstrumentChannel

from qblox_instruments import (
    SequencerStatus,
    SequencerStatuses,
    SequencerStatusFlags,
    SequencerStates,
    )


from .q1core import Q1Core
from .rt_renderer import Renderer, MockDataEntry


logger = logging.getLogger(__name__)

MockDataType = Iterable[MockDataEntry]


class Q1Sequencer(InstrumentChannel):

    # only logged
    _seq_log_only_parameters = [
        # -- only printed:
        'sync_en',
        'marker_ovr_en',
        'marker_ovr_value',
        'cont_mode_en_awg_path0',
        'cont_mode_en_awg_path1',
        'cont_mode_waveform_idx_awg_path0',
        'cont_mode_waveform_idx_awg_path1',
        'upsample_rate_awg_path0',
        'upsample_rate_awg_path1',
        'nco_freq_cal_type_default',
        ]
    _seq_log_only_parameters_qrm = [
        'connect_acq_I',
        'connect_acq_Q',
        'nco_prop_delay_comp',
        'nco_prop_delay_comp_en',
        'ttl_acq_threshold',
        'ttl_acq_input_select',
        ]
    _seq_log_only_parameters_qrm_rf = [
        'connect_acq',
        'nco_prop_delay_comp',
        'nco_prop_delay_comp_en',
        ]

    def __init__(self, parent, name, sim_type):
        super().__init__(parent, name)
        self._is_qcm = sim_type in ['QCM', 'QCM-RF', 'Viewer']
        self._is_qrm = sim_type in ['QRM', 'QRM-RF', 'Viewer']
        self._is_rf = sim_type in ['QCM-RF', 'QRM-RF']

        if self._is_qrm:
            if self._is_rf:
                log_params = self._seq_log_only_parameters + self._seq_log_only_parameters_qrm_rf
            else:
                log_params = self._seq_log_only_parameters + self._seq_log_only_parameters_qrm
            self._v_max = 0.5
        else:
            log_params = self._seq_log_only_parameters
            self._v_max = 2.5

        if sim_type == 'Viewer':
            self._v_max = 2.5

        if self._is_rf:
            self._v_max = 3.3

        for par_name in log_params:
            self.add_parameter(par_name,
                               set_cmd=partial(self._log_set, par_name))

        self.add_parameter('nco_phase_offs', set_cmd=self._nco_phase_offs)
        for i in range(2):
            self.add_parameter(f'gain_awg_path{i}', set_cmd=partial(self._gain_awg, path=i))
            self.add_parameter(f'offset_awg_path{i}', set_cmd=partial(self._offset_awg, path=i))

        self.add_parameter('sequence', set_cmd=self.upload)
        self.add_parameter('mod_en_awg', set_cmd=self._set_mod_en_awg)
        self.add_parameter('nco_freq', set_cmd=self._set_nco_freq)
        self.add_parameter('mixer_corr_gain_ratio', set_cmd=self._set_mixer_gain_ratio)
        self.add_parameter('mixer_corr_phase_offset_degree', set_cmd=self._set_mixer_phase_offset_degree)

        n_out_ch = 4 if self._is_qcm else 2
        if self._is_rf:
            n_out_ch //= 2
        for i in range(n_out_ch):
            self.add_parameter(f'connect_out{i}',
                               set_cmd=partial(self._connect_out, i))

        for i in range(1, 16):
            self.add_parameter(f'trigger{i}_count_threshold',
                               set_cmd=partial(self._set_trigger_count_threshold, i))
            self.add_parameter(f'trigger{i}_threshold_invert',
                               set_cmd=partial(self._set_trigger_threshold_invert, i))

        if self._is_qrm:
            self.add_parameter('demod_en_acq', set_cmd=self._set_demod_en_acq)
            self.add_parameter('integration_length_acq', set_cmd=self._set_integratrion_length_acq)
            self.add_parameter('thresholded_acq_rotation', set_cmd=self._set_thresholded_acq_rotation)
            self.add_parameter('thresholded_acq_threshold', set_cmd=self._set_thresholded_acq_threshold)
            self.add_parameter('thresholded_acq_trigger_en', set_cmd=self._set_thresholded_acq_trigger_en)
            self.add_parameter('thresholded_acq_trigger_address', set_cmd=self._set_thresholded_acq_trigger_address)
            self.add_parameter('thresholded_acq_trigger_invert', set_cmd=self._set_thresholded_acq_trigger_invert)
            self.add_parameter('ttl_acq_auto_bin_incr_en', set_cmd=self._set_ttl_acq_auto_bin_incr_en)

        self._trace = False
        self.reset()

    def config(self, name, value):
        if name == 'max_render_time':
            self.rt_renderer.max_render_time = value
        elif name == 'max_core_cycles':
            self.q1core.max_core_cycles = value
        elif name == 'trace':
            self._trace = value
            self.rt_renderer.trace_enabled = value
        elif name == 'render_repetitions':
            self.q1core.skip_loops = ("_start", ) if not value else ()
        elif name == 'skip_loops':
            self.q1core.skip_loops = value
        elif name == 'skip_wait_sync':
            self.rt_renderer.skip_wait_sync = value
        elif name == 'acq_trigger_value':
            self.rt_renderer.acq_trigger_value = value

    def reset(self):
        self.waveforms = {}
        self.weights = {}
        self.acquisitions = {}
        self._mock_data = {}
        self._trigger_events = []
        self._scope_data = {}
        self.output_selected_path = ['off'] * 4
        self._paths_used = [False, False]
        self.run_state = 'IDLE'
        self.rt_renderer = Renderer(self.name)
        self.rt_renderer.trace_enabled = self._trace
        self.q1core = Q1Core(self.name, self.rt_renderer, self._is_qrm)
        self.reset_trigger_thresholding()

    def get_simulation_end_time(self):
        return self.rt_renderer.time

    def _log_set(self, name, value):
        logger.info(f'{self.name}: {name}={value}')

    def _nco_phase_offs(self, degrees):
        self.rt_renderer.set_ph((degrees / 360) % 1 * 1e9)

    def _gain_awg(self, gain, path):
        value = int(gain*32767)
        self.rt_renderer.gain_awg_path(value, path)

    def _offset_awg(self, offset, path):
        value = int(offset*32767)
        self.rt_renderer.offset_awg_path(value, path)

    def _set_mod_en_awg(self, value):
        logger.debug(f'{self.name}: mod_en_awg={value}')
        self.rt_renderer.mod_en_awg = value

    def _set_nco_freq(self, value):
        logger.info(f'{self.name}: nco_freq={value}')
        self.rt_renderer.nco_frequency = value

    def _set_demod_en_acq(self, value):
        logger.debug(f'{self.name}: demod_en_acq={value}')
        self.rt_renderer.demod_en_acq = value

    def _set_mixer_gain_ratio(self, value):
        logger.debug(f'{self.name}: mixer_gain_ratio={value}')
        self.rt_renderer.mixer_gain_ratio = value

    def _set_mixer_phase_offset_degree(self, value):
        logger.debug(f'{self.name}: mixer_phase_offset_degree={value}')
        self.rt_renderer.mixer_phase_offset_degree = value

    def _connect_out(self, out, value):
        logger.debug(f'{self.name}: _connect_out{out}={value}')
        self.output_selected_path[out] = value
        paths_used = [False, False]
        for value in self.output_selected_path:
            if "I" in value:
                paths_used[0] = True
            if "Q" in value:
                paths_used[1] = True
        self._paths_used = paths_used
        self.rt_renderer.enable_paths(self._paths_used)

    def _set_trigger_count_threshold(self, address, count):
        self.rt_renderer.set_trigger_count_threshold(address, count)

    def _set_trigger_threshold_invert(self, address, invert: bool):
        self.rt_renderer.set_trigger_threshold_invert(address, invert)

    def _set_integratrion_length_acq(self, value):
        self.rt_renderer.set_integratrion_length_acq(value)

    def _set_thresholded_acq_rotation(self, value):
        self.rt_renderer.set_thresholded_acq_rotation(value)

    def _set_thresholded_acq_threshold(self, value):
        self.rt_renderer.set_thresholded_acq_threshold(value)

    def _set_thresholded_acq_trigger_en(self, value):
        self.rt_renderer.set_thresholded_acq_trigger_en(value)

    def _set_thresholded_acq_trigger_address(self, value):
        self.rt_renderer.set_thresholded_acq_trigger_addr(value)

    def _set_thresholded_acq_trigger_invert(self, value):
        self.rt_renderer.set_thresholded_acq_trigger_invert(value)

    def _set_ttl_acq_auto_bin_incr_en(self, value):
        self.rt_renderer.set_ttl_acq_auto_bin_incr_en(value)

    def upload(self, sequence):
        if isinstance(sequence, dict):
            pdict = sequence
        else:
            filename = sequence
            with open(filename) as fp:
                pdict = json.load(fp)
        program = pdict['program']
        self.q1core.load(program)
        waveforms = pdict['waveforms']
        self._set_waveforms(waveforms)
        if self._is_qrm:
            weights = pdict['weights']
            acquisitions = pdict['acquisitions']
            self._set_weights(weights)
            self._set_acquisitions(acquisitions)

    def update_sequence(self, erase_existing: bool = False, **sequence):
        if "program" in sequence:
            self.q1core.load(sequence["program"])

        if erase_existing:
            if "waveforms" in sequence:
                self._set_waveforms(sequence["waveforms"])
            if "weights" in sequence:
                self._set_weights(sequence["weights"])
            if "acquisitions" in sequence:
                self._set_acquisitions(sequence["acquisitions"])
        else:
            if "waveforms" in sequence:
                waveforms = self.waveforms
                indices = {wvf["index"]: name for name, wvf in waveforms.items()}
                for name, waveform in sequence["waveforms"].items():
                    index = waveform["index"]
                    if index in indices:
                        del waveforms[indices[index]]
                    waveforms[name] = waveform
                self._set_waveforms(waveforms)
            if "weights" in sequence:
                weights = self.weights
                indices = {weight["index"]: name for name, weight in weights.items()}
                for name, weight in sequence["weights"].items():
                    index = weight["index"]
                    if index in indices:
                        del weights[indices[index]]
                    weights[name] = weight
                self._set_weights(weights)
            if "acquisitions" in sequence:
                acquisitions = self.acquisitions
                indices = {acq["index"]: name for name, acq in acquisitions.items()}
                for name, acquisition in sequence["acquisitions"].items():
                    index = acquisition["index"]
                    if index in indices:
                        del acquisitions[indices[index]]
                    acquisitions[name] = acquisition
                self._set_acquisitions(acquisitions)

    def _set_waveforms(self, waveforms):
        self.waveforms = waveforms.copy()
        wavedict = {}
        for name, datadict in waveforms.items():
            index = int(datadict['index'])
            data = np.array(datadict['data'])
            wavedict[index] = data
        self.rt_renderer.set_waveforms(wavedict)

    def _set_weights(self, weights):
        self.weights = weights.copy()
        weightsdict = {}
        for name, datadict in weights.items():
            index = int(datadict['index'])
            data = np.array(datadict['data'])
            weightsdict[index] = data
        self.rt_renderer.set_weights(weightsdict)

    def _set_acquisitions(self, acquisitions):
        self.acquisitions = acquisitions.copy()
        acq_dict = {}
        for name, datadict in acquisitions.items():
            index = int(datadict['index'])
            num_bins = int(datadict['num_bins'])
            acq_dict[index] = num_bins
        self.rt_renderer.set_acquisitions(acq_dict)

    def _set_rt_mock_data(self):
        for name, md in self._mock_data.items():
            if name not in self.acquisitions:
                logger.warning(f"no acquisition for mock_data '{name}'")
                continue
            try:
                data = np.asarray(next(md))
            except StopIteration:
                raise Exception('No more mock data')
            bin_num = int(self.acquisitions[name]['index'])
            self.rt_renderer.set_mock_data(bin_num, data)

    def get_sequencer_status(self, timeout: int = 0, timeout_poll_res: float = 0.02):
        info_flags = []
        warn_flags = []
        error_flags = [
            SequencerStatusFlags[flag.replace(' ', '_')]
            for flag in self.q1core.errors | self.rt_renderer.errors
            ]
        log = []
        if self._is_qrm:
            info_flags.append(SequencerStatusFlags.ACQ_BINNING_DONE)

        return SequencerStatus(
            SequencerStatuses.OKAY,
            SequencerStates[self.run_state],
            info_flags,
            warn_flags,
            error_flags,
            log,
        )

    def get_acquisition_status(self, timeout: int = 0, timeout_poll_res: float = 0.02, check_seq_state: bool = True):
        if not self._is_qrm:
            raise NotImplementedError('Instrument type is not QRM')
        return True

    def get_waveforms(self, *, as_numpy: bool = False):
        return {
            name: {
                "data": np.asarray(wvf["data"]) if as_numpy else list(wvf["data"]),
                "index": wvf["index"],
                }
            for name, wvf in self.waveforms.items()
            }

    def get_weights(self, *, as_numpy: bool = False):
        if not self._is_qrm:
            raise NotImplementedError('Instrument type is not QRM')
        return {
            name: {
                "data": np.asarray(weight["data"]) if as_numpy else list(weight["data"]),
                "index": weight["index"],
                }
            for name, weight in self.weights.items()
            }

    def set_trigger_thresholding(self, address: int, count: int, invert: bool) -> None:
        self.rt_renderer.set_trigger_count_threshold(address, count)
        self.rt_renderer.set_trigger_threshold_invert(address, invert)

    def get_trigger_thresholding(self, address: int) -> tuple[int, bool]:
        return (
            self.rt_renderer.get_trigger_count_threshold(address),
            self.rt_renderer.get_trigger_threshold_invert(address)
            )

    def reset_trigger_thresholding(self) -> None:
        for i in range(1, 16):
            self.set_trigger_thresholding(i, 1, False)

    def sideband_cal(self) -> None:
        logger.info(f"Calibrate sideband {self.name}")

    def arm_sequencer(self):
        self.run_state = 'ARMED'

    def start_sequencer(self):
        self.run()

    def run(self):
        self.run_state = 'RUNNING'
        self.rt_renderer.reset()
        self._set_rt_mock_data()
        self.rt_renderer.trigger_events = self._trigger_events
        self.q1core.run()
        self.run_state = 'STOPPED'

    def get_acquisitions(self, *, as_numpy: bool = False):
        if not self._is_qrm:
            raise NotImplementedError('Instrument type is not QRM')
        cnt, data, thresholded = self.rt_renderer.get_acquisition_data()
        result = {}
        for name, datadict in self.acquisitions.items():
            index = int(datadict['index'])
            acq_count = cnt[index]
            path_data = data[index]/acq_count[:, None]
            threshold = thresholded[index]/acq_count
            path0 = path_data[:, 0]
            path1 = path_data[:, 1]

            if not as_numpy:
                acq_count = acq_count.tolist()
                threshold = threshold.tolist()
                path0 = path0.tolist()
                path1 = path1.tolist()

            result[name] = {
                'index': index,
                'acquisition': {
                    'bins': {
                        'integration': {
                            'path0': path0,
                            'path1': path1,
                            },
                        'threshold': threshold,
                        'avg_cnt': acq_count,
                    }
                }}
            if name in self._scope_data:
                scope_data = self._scope_data[name]
                result[name]['acquisition']["scope"] = {
                    'path0': {
                        'data': scope_data['path0'],
                        'avg_cnt': scope_data['avg_cnt'],
                        },
                    'path1': {
                        'data': scope_data['path1'],
                        'avg_cnt': scope_data['avg_cnt'],
                        },
                    }
        return result

    def delete_acquisition_data(self, name='', all=False):
        if all:
            self.rt_renderer.delete_acquisition_data_all()
            self._scope_data = {}
        else:
            index = self.acquisitions[name]['index']
            self.rt_renderer.delete_acquisition_data(index)
            if name in self._scope_data:
                del self._scope_data[name]

    def store_scope_acquisition(self, acq_name):
        self._scope_data[acq_name] = {
            "path0": np.cos(np.arange(131072)/1000*2*np.pi),
            "path1": np.sin(np.arange(131072)/1000*2*np.pi),
            "avg_cnt": 1,
            }

    # --- Simulator specific methods ---

    def set_acquisition_mock_data(self,
                                  data: Iterable[MockDataType] | None,
                                  name='default',
                                  repeat=False):
        '''
        Sets mock acquisition data for 1 or more runs of the sequence.

        `data` is a list with lists of values to use per run of the sequence.
        The list for a run should have a length equal or bigger than
        the number of acquire calls in the executed sequence.
        The entry for an acquire call is used for both paths.
        If it is a single float value then it is used for both paths.
        If it is a complex value then the real part is used for path 0 and
        the imaginary part for path 1.
        If it is a sequence of two floats then the first is used for path 0
        and the second for path 1.

        Args:
            data:
                if None clears the data and resets default behavior,
                otherwise list of mock data for every run of the sequence.
            name: name of the acquisition
            repeat:
                if True repeatly cycles through the list of mock data,
                otherwise an exception is raised when the list of mock data is exhausted.

        Example:
            # set data for 1 run to return the values 0 till 19 on path 0 and path 1
            data = [np.arange(20)]
            sim.sequencers[0].set_acquisition_mock_data(data)

            # set data for every run to return IQ values with changing phase
            # on path 0 and 1
            data = [np.exp(np.pi*1j*np.arange(20)/10)]
            sim.sequencers[0].set_acquisition_mock_data(data, repeat=True)

            # set data for every run to return the values 0 till 19 on path 0
            # and 100 till 119 on path 1.
            data = [np.arange(20) + 1j*np.arange(100,120)]
            sim.sequencers[0].set_acquisition_mock_data(data, repeat=True)

            # set data for 2 runs to return the values 0 till 19 on the first run
            # and 100 till 119 on the second run.
            data2 = [np.arange(20), np.arange(100, 120)]
            sim.sequencers[0].set_acquisition_mock_data(data2)

            # Return 0...19 and 100...119 alternatingly.
            sim.sequencers[0].set_acquisition_mock_data(data2, repeat=True)

            # reset default behaviour.
            sim.sequencers[0].set_acquisition_mock_data(None)
        '''
        if data is None and name in self._mock_data:
            del self._mock_data[name]
        else:
            self._mock_data[name] = MockData(data, repeat)

    def get_used_triggers(self):
        return self.q1core.get_used_triggers()

    def get_acq_trigger_events(self):
        return self.rt_renderer.acq_trigger_events

    def set_trigger_events(self, events):
        self._trigger_events = events

    def set_forced_condition_value(self, value):
        self.rt_renderer.forced_condition_value = value

    def get_enabled_outputs(self):
        outputs = ""
        if self._paths_used[0]:
            outputs += "I"
        if self._paths_used[1]:
            outputs += "Q"
        return outputs

    def get_output(self, t_min=None, t_max=None, analogue_filter=False, output_frequency=4e9):
        return self.rt_renderer.get_output(
            self._v_max,
            t_min=t_min,
            t_max=t_max,
            analogue_filter=analogue_filter,
            output_frequency=output_frequency,
            )

    def print_registers(self, reg_nrs=None):
        self.q1core.print_registers(reg_nrs)


@dataclass
class MockData:
    data: Iterable[MockDataType]
    repeat: bool
    data_iter: Iterator[MockDataType] = None

    def __post_init__(self):
        self.data_iter = iter(self.data)

    def __next__(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            if not self.repeat:
                raise
            self.data_iter = iter(self.data)
            return next(self.data_iter)
