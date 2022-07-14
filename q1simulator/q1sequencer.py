import logging
from functools import partial
import json
import numpy as np

from qcodes.instrument.channel import InstrumentChannel

from .q1core import Q1Core
from .rt_renderer import Renderer

from qblox_instruments import SequencerState, SequencerStatus, SequencerStatusFlags


class Q1Sequencer(InstrumentChannel):

    # only logged
    _seq_log_only_parameters = [
        # -- only printed:
        'sync_en',
        'nco_phase_offs',
        'marker_ovr_en',
        'marker_ovr_value',
        'cont_mode_en_awg_path0',
        'cont_mode_en_awg_path1',
        'cont_mode_waveform_idx_awg_path0',
        'cont_mode_waveform_idx_awg_path1',
        'upsample_rate_awg_path0',
        'upsample_rate_awg_path1',
        'gain_awg_path0',
        'gain_awg_path1',
        'offset_awg_path0',
        'offset_awg_path1',
        ]
    _seq_log_only_parameters_qrm = [
        'integration_length_acq',
        'phase_rotation_acq',
        'discretization_threshold_acq',
        ]

    def __init__(self, parent, name, sim_type):
        super().__init__(parent, name)
        self._is_qcm = sim_type in ['QCM', 'QCM-RF', 'Viewer']
        self._is_qrm = sim_type in ['QRM', 'QRM-RF', 'Viewer']

        if self._is_qrm:
            log_params = self._seq_log_only_parameters + self._seq_log_only_parameters_qrm
        else:
            log_params = self._seq_log_only_parameters

        for par_name in log_params:
            self.add_parameter(par_name,
                               set_cmd=partial(self._log_set, par_name))

        self.add_parameter('sequence', set_cmd=self.upload)
        self.add_parameter('mod_en_awg', set_cmd=self._set_mod_en_awg)
        self.add_parameter('nco_freq', set_cmd=self._set_nco_freq)
        self.add_parameter('mixer_corr_gain_ratio', set_cmd=self._set_mixer_gain_ratio)
        self.add_parameter('mixer_corr_phase_offset_degree', set_cmd=self._set_mixer_phase_offset_degree)

        self.add_parameter('channel_map_path0_out0_en',
                           set_cmd=partial(self._set_channel_map_path_en, 0, 0))
        self.add_parameter('channel_map_path1_out1_en',
                           set_cmd=partial(self._set_channel_map_path_en, 1, 1))
        if self._is_qcm:
            self.add_parameter('channel_map_path0_out2_en',
                               set_cmd=partial(self._set_channel_map_path_en, 0, 2))
            self.add_parameter('channel_map_path1_out3_en',
                               set_cmd=partial(self._set_channel_map_path_en, 1, 3))
        if self._is_qrm:
            self.add_parameter('demod_en_acq', set_cmd=self._set_demod_en_acq)

        self.reset()

    def config(self, name, value):
        if name == 'name':
            self.name = value
            self.rt_renderer.name = value
        elif name == 'max_render_time':
            self.rt_renderer.max_render_time = value
        elif name == 'max_core_cycles':
            self.q1core.max_core_cycles = value

    def reset(self):
        self.waveforms = {}
        self.weights = {}
        self.acquisition_bins = {}

        self.run_state = 'IDLE'
        self.rt_renderer = Renderer(self.name)
        self.q1core = Q1Core(self.name, self.rt_renderer, self._is_qrm)

    def _log_set(self, name, value):
        logging.info(f'{self.name}: {name}={value}')

    def _set_mod_en_awg(self, value):
        logging.debug(f'{self.name}: mod_en_awg={value}')
        self.rt_renderer.mod_en_awg = value

    def _set_nco_freq(self, value):
        logging.info(f'{self.name}: nco_freq={value}')
        self.rt_renderer.nco_frequency = value

    def _set_demod_en_acq(self, value):
        logging.debug(f'{self.name}: demod_en_acq={value}')
        self.rt_renderer.demod_en_acq = value

    def _set_mixer_gain_ratio(self, value):
        logging.debug(f'{self.name}: mixer_gain_ratio={value}')
        self.rt_renderer.mixer_gain_ratio = value

    def _set_mixer_phase_offset_degree(self, value):
        logging.debug(f'{self.name}: mixer_phase_offset_degree={value}')
        self.rt_renderer.mixer_phase_offset_degree = value

    def _set_channel_map_path_en(self, path, out, value):
        logging.debug(f'{self.name}: channel_map_path{path}_out{out}_en={value}')
        self.rt_renderer.path_enable(path, out, value)

    def upload(self, file_name):
        with open(file_name) as fp:
            pdict = json.load(fp)
        waveforms = pdict['waveforms']
        weights = pdict['weights']
        acquisitions = pdict['acquisitions']
        program = pdict['program']
        self._set_waveforms(waveforms)
        self._set_weights(weights)
        self._set_acquisition_bins(acquisitions)
        self.q1core.load(program)

    def _set_waveforms(self, waveforms):
        self.waveforms = waveforms
        wavedict = {}
        for name, datadict in waveforms.items():
            index = int(datadict['index'])
            data = np.array(datadict['data'])
            wavedict[index] = data
        self.rt_renderer.set_waveforms(wavedict)

    def _set_weights(self, weights):
        self.weights = weights
        weightsdict = {}
        for name, datadict in weights.items():
            index = int(datadict['index'])
            data = np.array(datadict['data'])
            weightsdict[index] = data
        self.rt_renderer.set_weights(weightsdict)

    def _set_acquisition_bins(self, acq_bins):
        self.acquisition_bins = acq_bins
        bins_dict = {}
        for name, datadict in acq_bins.items():
            index = int(datadict['index'])
            num_bins = int(datadict['num_bins'])
            bins_dict[index] = num_bins
        self.rt_renderer.set_acquisition_bins(bins_dict)

    def get_state(self):
        flags = list(self.q1core.errors | self.rt_renderer.errors)
        return SequencerState(
            SequencerStatus[self.run_state],
            [SequencerStatusFlags[flag.replace(' ','_')] for flag in flags],
        )

    def get_acquisition_state(self):
        if not self._is_qrm:
            raise NotImplementedError('Instrument type is not QRM')
        return True

    def arm(self):
        self.run_state = 'ARMED'

    def run(self):
        self.run_state = 'RUNNING'
        self.rt_renderer.reset()
        self.q1core.run()
        self.run_state = 'STOPPED'

    def get_acquisition_data(self):
        if not self._is_qrm:
            raise NotImplementedError('Instrument type is not QRM')
        cnt,data = self.rt_renderer.get_acquisition_data()
        result = {}
        for name, datadict in self.acquisition_bins.items():
            index = int(datadict['index'])
            num_bins = int(datadict['num_bins'])
            acq_count = cnt[index]
            path_data = [d/c if c > 0 else float('nan')
                         for d,c in zip(data[index], cnt[index])]

            result[name] = {
                'index':index,
                'acquisition':{
                    'bins':{
                        'integration': {
                            'path0':path_data,
                            'path1':path_data,
                            },
                        'threshold':[0.0]*num_bins,
                        'avg_cnt':acq_count,
                    }
                }}
        return result

    def plot(self):
        self.rt_renderer.plot()

    def print_registers(self, reg_nrs=None):
        self.q1core.print_registers(reg_nrs)
