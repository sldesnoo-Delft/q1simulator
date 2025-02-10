import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Sequence, Iterable
from numbers import Number
from collections.abc import Sequence as AbcSequence
from functools import wraps

import numpy as np
import matplotlib.pyplot as pt

from .triggers import TriggerEvent
from .analogue_filter import AnalogueFilter


logger = logging.getLogger(__name__)

MockDataEntry = float | complex | Sequence[float]


@dataclass
class Settings:
    marker: int = 0
    awg_offs: np.ndarray = field(default_factory=lambda: np.zeros(2, np.int16))
    awg_gain: np.ndarray = field(default_factory=lambda: np.full(2, 32767, np.int16))
    reset_phase: bool = False
    relative_phase: float | None = None
    phase_shift: float = 0
    frequency: float | None = None


@dataclass
class AcqConf:
    length: int = 0  # max 16 ms
    rotation: float = 0.0  # deg.
    threshold: float = 0.0
    trigger_en: bool = False
    trigger_addr: int = 0
    trigger_invert: bool = False


def _phase2float(phase_uint32):
    # phase in rotations, i.e. unit = 2 pi rad
    phase = phase_uint32/1e9
    # print(f'Phase {phase:9.6f} rotations ({phase*360:8.3f} deg)')
    return phase


def _freq2Hz(freq_uint32):
    # freq in Hz
    # convert uint to int32 if value > 2**31
    res = np.int32(freq_uint32) / 4
    return res


def float2int16array(value):
    # TODO: check on float < -1.0 or > +1.0
    # Scale to 16 bit value, but store in 32 bit to avoid
    # overflow on later operations.
    return np.array(value*2**15).astype(np.int32)


def _i16(value):
    return np.int32(value).astype(np.int16)


def check_conditional(func):
    @wraps(func)
    def func_wrapper(self, *args, **kwargs):
        if self.skip_rt:
            self._else_wait()
        else:
            func(self, *args, **kwargs)

    return func_wrapper


class Renderer:

    def __init__(self, name):
        self.name = name
        self._max_render_time = 2_000_000
        self.wavedict_float = {}
        self.wavedict = {}
        self.acq_weights = {}
        self.acq_bins = {}
        self.output_selected_path = ['off'] * 4
        self.nco_frequency = 0.0
        self.mod_en_awg = False
        self.mixer_gain_ratio = 1.0
        self.mixer_phase_offset_degree = 0.0
        self.delete_acquisition_data_all()
        self.next_settings = Settings()
        self.reset()
        self.trace_enabled = False
        self.skip_wait_sync = True
        self.acq_trigger_value = None
        self.threshold_count = np.full(15, 0, dtype=np.uint16)
        self.threshold_invert = np.zeros(15, dtype=bool)
        self.acq_conf = AcqConf()

    def reset(self):
        # start with the old settings / values set via qcodes.
        self.settings = deepcopy(self.next_settings)
        self.time = 0
        # Qblox sequencer has 3 different phase registers
        self.nco_phase_offset = 0.0
        self.relative_phase = 0.0
        self.delta_phase = 0.0
        self.wave_start = 0
        self.waves_end = (0, 0)
        self.waves = (None, None)
        self.out0 = []
        self.out1 = []
        self.marker_out = [list() for _ in range(4)]  # a list per marker
        self.acq_times = {i: [] for i in self.acq_bins}
        self.acq_buffer = AcqBuffer()
        self.acq_trigger_events = []
        self.mock_data = {}
        self.errors = set()
        self.latch_enabled = False
        self.latch_regs = np.zeros(15, dtype=np.uint16)
        self.trigger_events: list[TriggerEvent] = []
        self.skip_rt = False
        self.else_wait = 0

    def gain_awg_path(self, gain, path):
        self.next_settings.awg_gain[path] = gain

    def offset_awg_path(self, offset, path):
        self.next_settings.awg_offs[path] = offset

    def connect_out(self, out, value):
        self.output_selected_path[out] = value

    def set_waveforms(self, wavedict):
        self.wavedict_float = wavedict
        self.wavedict = {
            key: float2int16array(value)
            for key, value in wavedict.items()
        }

    def set_weights(self, weightsdict):
        self.acq_weights = weightsdict

    def set_acquisition_bins(self, acq_bins):
        self.acq_bins = acq_bins
        self.delete_acquisition_data_all()

    def set_trigger_count_threshold(self, addr, count):
        self.threshold_count[addr-1] = count

    def set_trigger_threshold_invert(self, addr, invert):
        self.threshold_invert[addr-1] = invert

    def get_trigger_count_threshold(self, addr):
        return self.threshold_count[addr-1]

    def get_trigger_threshold_invert(self, addr):
        return self.threshold_invert[addr-1]

    def set_integratrion_length_acq(self, value):
        self.acq_conf.length = value

    def set_thresholded_acq_rotation(self, value):
        self.acq_conf.rotation = value

    def set_thresholded_acq_threshold(self, value):
        self.acq_conf.threshold = value

    def set_thresholded_acq_trigger_en(self, value):
        self.acq_conf.trigger_en = value

    def set_thresholded_acq_trigger_addr(self, value):
        self.acq_conf.trigger_addr = value

    def set_thresholded_acq_trigger_invert(self, value):
        self.acq_conf.trigger_invert = value

    def set_mrk(self, value):
        self.next_settings.marker = value

    def set_freq(self, freq):
        self.next_settings.frequency = _freq2Hz(freq)

    def reset_ph(self):
        self.next_settings.reset_phase = True

    def set_ph(self, phase):
        self.next_settings.relative_phase = _phase2float(phase)

    def set_ph_delta(self, phase_delta):
        self.next_settings.phase_shift = _phase2float(phase_delta)

    def set_awg_gain(self, gain0, gain1):
        self.next_settings.awg_gain[:] = gain0, gain1

    def set_awg_offs(self, offset0, offset1):
        self.next_settings.awg_offs[:] = offset0, offset1

    @check_conditional
    def upd_param(self, wait_after):
        self._update_settings()
        self._render(wait_after)

    @check_conditional
    def play(self, wave0, wave1, wait_after):
        self._update_settings()
        if self.trace_enabled:
            self._trace(f'Play {wave0} {wave1}')
        if wave0 not in self.wavedict:
            self._error('AWG WAVE PLAYBACK INDEX INVALID PATH 0')
        elif wave1 not in self.wavedict:
            self._error('AWG WAVE PLAYBACK INDEX INVALID PATH 1')
        else:
            self.wave_start = self.time
            self.waves = (self.wavedict[wave0], self.wavedict[wave1])
            self.waves_end = (self.time + len(self.waves[0]),
                              self.time + len(self.waves[1]))
        self._render(wait_after)

    @check_conditional
    def acquire(self, bins, bin_index, wait_after):
        self._update_settings()
        if self.trace_enabled:
            self._trace(f'Acquire {bins} {bin_index} ({self.acq_conf.length} ns)')
        self._add_acquisition(bins, bin_index, self.acq_conf.length)
        self._render(wait_after)

    @check_conditional
    def acquire_weighed(self, bins, bin_index, weight0, weight1, wait_after):
        self._update_settings()
        if self.trace_enabled:
            self._trace(f'AcquireWeighed {bins} {bin_index} {weight0} {weight1}')
        if weight0 not in self.acq_weights:
            self._error('ACQ WEIGHT PLAYBACK INDEX INVALID PATH 0')
        elif weight1 not in self.acq_weights:
            self._error('ACQ WEIGHT PLAYBACK INDEX INVALID PATH 1')
        else:
            duration = max(
                len(self.acq_weights[weight0]),
                len(self.acq_weights[weight1])
            )
            self._add_acquisition(bins, bin_index, duration)
        self._render(wait_after)

    @check_conditional
    def wait(self, time):
        if self.trace_enabled:
            self._trace(f'Wait {time}')
        self._render(time)

    def wait_sync(self, wait_after):
        if self.skip_wait_sync:
            return
        self._render(wait_after)

    def set_cond(self, enable, mask, op, else_wait):
        if not enable:
            self._trace('Cond disabled')
            self.skip_rt = False
            return
        self._process_triggers()
        # numpy arrays
        mask_ar = np.unpackbits([np.uint8(mask >> 8), np.uint8(mask & 0xFF)])[:0:-1]
        state = ((self.latch_regs >= self.threshold_count) ^ self.threshold_invert) & mask_ar
        bits_set = np.sum(state)
        bits_mask = np.sum(mask_ar)
        if op == 0:  # OR
            match = bits_set != 0
        elif op == 1:  # NOR
            match = bits_set == 0
        elif op == 2:  # AND
            match = bits_set == bits_mask
        elif op == 3:  # NAND
            match = bits_set != bits_mask
        elif op == 4:  # XOR
            match = (bits_set % 2) == 1
        elif op == 5:  # XNOR
            match = (bits_set % 2) == 0
        else:
            raise Exception(f'Unknown operator {op}')
        logger.debug(f'set_cond 1, {mask}, {op}, {else_wait}')
        logger.debug(f'latches: {self.latch_regs}')
        logger.debug(f'mask:    {mask_ar}')
        logger.debug(f'state:   {state}')
        logger.debug(f'cond:    {match}')
        self._trace(f'Cond {match} {state}')
        self.skip_rt = not match
        self.else_wait = else_wait

    def _else_wait(self, *args, **kwargs):
        self._render(self.else_wait)

    def set_latch_en(self, enable, wait_after):
        if self.trace_enabled:
            self._trace(f'Latch {"on" if enable else "off"}')
        self._process_triggers()
        self.latch_enabled = enable
        self._render(wait_after)

    def latch_rst(self, wait):
        if self.trace_enabled:
            self._trace(f'Latch reset {wait}')
        self._process_triggers()
        self.latch_regs[:] = 0
        self._render(wait)

    def sim_trigger(self, addr, value):
        # addresses 1..15. Register index (bits) 0..14!
        index = int(addr)-1
        self.latch_regs[index] += int(value)

    @property
    def max_render_time(self):
        return self._max_render_time

    @max_render_time.setter
    def max_render_time(self, value):
        self._max_render_time = int(value)

    def _error(self, msg):
        logger.error(f'{self.name}: {msg}')
        self.errors.add(msg)

    def _update_settings(self):
        new = self.next_settings
        old = self.settings

        if self.trace_enabled:
            msg = []
            if not np.array_equal(new.awg_gain, old.awg_gain):
                msg.append(f'awg_gain {new.awg_gain}')
            if not np.array_equal(new.awg_offs, old.awg_offs):
                msg.append(f'awg_offset {new.awg_offs}')
            if new.reset_phase:
                msg.append('reset_phase')
            if new.relative_phase is not None:
                msg.append(f'relative phase {new.relative_phase}')
            if new.phase_shift != 0.0:
                msg.append(f'add delta phase {new.phase_shift}')
            if new.marker != old.marker:
                msg.append(f'marker {new.marker:04b}')
            if len(msg) > 0:
                self._trace('Update: ' + '; '.join(msg))

        if new.reset_phase:
            self.nco_phase_offset = (-self.time * self.nco_frequency * 1e-9) % 1
            new.reset_phase = False
            # reset also resets the 2 other phase registers.
            self.relative_phase = 0.0
            self.delta_phase = 0.0
        if new.relative_phase is not None:
            self.relative_phase = new.relative_phase
            new.relative_phase = None
        if new.frequency is not None:
            phase = (self.nco_phase_offset + self.time * self.nco_frequency * 1e-9) % 1
            new_phase_offset = phase - (self.time * new.frequency * 1e-9) % 1
            self.nco_phase_offset = new_phase_offset
            self.nco_frequency = new.frequency
            new.frequency = None
        self.delta_phase += new.phase_shift
        new.phase_shift = 0.0
        if new.marker != old.marker:
            self._render_marker(old.marker, new.marker)
        # copy marker, offset and gain
        self.settings = deepcopy(self.next_settings)

    def _render_marker(self, old_marker, new_marker):
        for i in range(4):
            m = 1 << i
            m_old = (old_marker & m) != 0
            m_new = (new_marker & m) != 0
            if m_new != m_old:
                l = self.marker_out[i]
                if self.time < self.max_render_time:
                    l += [[self.time, m_old], [self.time, m_new]]
                elif l[-1][0] < self.max_render_time:
                    # add final marker step
                    l += [[self.max_render_time, m_old], [self.max_render_time, 0]]

    def _render(self, time):
        if time < 4:
            logger.error(f'{self.name}: wait_time ({time} ns) must be >= 4 ns')
            self._error('WAIT TIME < 4 ns')

        if time & 0x0003:
            logger.error(f'{self.name}: wait time not aligned on '
                         f'4 ns boundary: {time} ns (offset={time & 0x03} ns)')
            self._error('TIME NOT ALIGNED')

        # 16 bits, 4 ns resolution
        time &= 0xFFFC
        t_start = self.time
        t_end = t_start + time
        self.time = t_end

        # stop rendering when there is too much data
        if t_start > self.max_render_time:
            return
        if t_end > self.max_render_time:
            t_end = int(self.max_render_time)

        t_render = t_end - t_start

        s = self.settings

        path = np.zeros((2, t_render), dtype=np.int16)
        path[0] = s.awg_offs[0]
        path[1] = s.awg_offs[1]

        for i in range(2):
            if self.waves_end[0] > t_start:
                end = min(self.waves_end[i], t_end)
                data = self.waves[i][t_start-self.wave_start:end-self.wave_start]
                path[i][0:len(data)] += (s.awg_gain[i] * data) >> 15

        if self.mod_en_awg:
            t = np.arange(t_start, t_end)
            phase_offset = self.relative_phase + self.delta_phase
            if self.nco_frequency < 0:
                phase_offset = -phase_offset
            phase = 2*np.pi*(phase_offset + self.nco_phase_offset + self.nco_frequency * 1e-9 * t)
            nco = np.exp(1j*phase)
            # Multiplication factor as specified when modulating.
            nco *= np.sqrt(.5)
            data0 = nco.real*path[0] - nco.imag*path[1]
            if self.mixer_phase_offset_degree != 0.0:
                phase_offset = self.mixer_phase_offset_degree/180*np.pi
                nco *= np.exp(1j*phase_offset)
            data1 = nco.imag*path[0] + nco.real*path[1]
            if self.mixer_gain_ratio > 1.0:
                data0 *= 1/self.mixer_gain_ratio
            if self.mixer_gain_ratio < 1.0:
                data1 *= self.mixer_gain_ratio
            data0 = data0.astype(np.int16)
            data1 = data1.astype(np.int16)
        else:
            data0 = path[0]
            data1 = path[1]

        self.out0.append(data0)
        self.out1.append(data1)

    def _process_triggers(self):
        t = self.time
        # TODO Refactor trigger distribution. Request triggers for interval.
        while len(self.trigger_events) > 0 and self.trigger_events[0].time <= t:
            trigger = self.trigger_events.pop(0)
            if self.latch_enabled:
                index = trigger.addr-1
                self.latch_regs[index] += trigger.state
                self._trace(f'Latch reg {index} {trigger.state:+d} -> {self.latch_regs[index]}')

    def _get_acq_data(self, bins, default):
        mock_data_iter = self.mock_data.get(bins, None)
        if mock_data_iter is None:
            return (default, default)
        else:
            try:
                value = next(mock_data_iter)
                if isinstance(value, complex):
                    return (value.real, value.imag)
                if isinstance(value, Number):
                    return (value, value)
                if isinstance(value, (AbcSequence, np.ndarray)):
                    return (value[0], value[1])
                raise ValueError(f"Incompatible mock data {value}")
            except StopIteration:
                self._error('OUT OF MOCK DATA')
                return (np.nan, np.nan)

    def delete_acquisition_data_all(self):
        self.acq_count = {}
        self.acq_data = {}
        self.acq_thresholded = {}
        for index in self.acq_bins:
            self.delete_acquisition_data(index)

    def delete_acquisition_data(self, index):
        num_bins = self.acq_bins[index]
        self.acq_count[index] = np.zeros(num_bins, dtype=int)
        self.acq_data[index] = np.full((num_bins, 2), np.nan)
        self.acq_thresholded[index] = np.full(num_bins, np.nan)

    def _add_acquisition(self, bins, bin_index, duration):
        t = self.time
        if bins not in self.acq_bins:
            self._error('ACQ INDEX INVALID')
            return
        self.acq_times[bins].append((t, bin_index))
        if bin_index >= self.acq_bins[bins]:
            self._error('ACQ BIN INDEX INVALID')
            return
        if not self.acq_buffer.add(t):
            self._error('ACQ BINNING FIFO ERROR')
            return
        acq_conf = self.acq_conf
        value = self._get_acq_data(bins, t/1e6)
        angle = acq_conf.rotation/180*np.pi
        rot_value = value[0]*np.cos(angle) + value[1]*np.sin(angle)
        state = rot_value >= acq_conf.threshold
        self._trace(f'Acq result {state}, {rot_value}, {acq_conf.threshold}')

        if self.acq_count[bins][bin_index] == 0:
            self.acq_data[bins][bin_index] = value
            self.acq_thresholded[bins][bin_index] = state
        else:
            self.acq_data[bins][bin_index] += value
            self.acq_thresholded[bins][bin_index] += state
        self.acq_count[bins][bin_index] += 1

        if acq_conf.trigger_en:
            t_end = t + duration
            if self.acq_trigger_value is None:
                trigger_state = state ^ acq_conf.trigger_invert
            else:
                trigger_state = self.acq_trigger_value
            # TODO also add to trigger events. Part of TriggerDistributor redesign.
            # FIXME: own trigger is not counted.
            self.acq_trigger_events.append(TriggerEvent(acq_conf.trigger_addr, t_end, trigger_state))
            self._trace(f'Trigger {acq_conf.trigger_addr} {t_end} {trigger_state}')

    def _trace(self, msg):
        if self.trace_enabled:
            print(f'{self.time:-6} {msg}')

    def plot(self, v_max, plot_label, t_min=None, t_max=None, analogue_filter=False):
        def time_window(out, t_min, t_max):
            if t_max is not None:
                out = out[:t_max]
            if t_min is not None:
                out = out[t_min:]
            return out

        if analogue_filter:
            _filter = AnalogueFilter("QCM")

        scaling = v_max/2**15
        t_end = self.time
        if self.time > self.max_render_time:
            max_ms = self.max_render_time / 1e6
            t_end = self.max_render_time
            print(f'{self.name}: Rendering truncated at {max_ms:3.1f} ms. Total time: {self.time/1e6:4.1f} ms')

        if t_min is not None:
            t = np.arange(t_min, min(t_max, t_end) if t_max is not None else t_end)
        elif analogue_filter:
            t = np.arange(0, min(t_max, t_end) if t_max is not None else t_end)

        n_ch_out = 0
        for value in self.output_selected_path:
            if value in ['I', 'Q']:
                n_ch_out += 1
            if value == 'IQ':
                n_ch_out += 2
        if n_ch_out == 0:
            print(f'No outputs enabled for {plot_label}')

        for i, value in enumerate(self.output_selected_path):
            label = plot_label
            if label == self.name:
                label += f'.out{i}'
            if value in ('I', 'IQ'):
                if n_ch_out > 1:
                    label += '-I'
                out0 = scaling * np.concatenate(self.out0)
                out0 = time_window(out0, t_min, t_max)
                # print(f'Average V: {np.mean(out0)*1000:5.2f} mV')
                if t_min is None and not analogue_filter:
                    pt.plot(out0, label=label)
                elif analogue_filter:
                    ta, outa = _filter.get_awg_output(t, out0)
                    pt.plot(ta, outa, label=label)
                else:
                    pt.plot(t, out0, label=label)
            if value in ('Q', 'IQ'):
                if n_ch_out > 1:
                    label += '-Q'
                out1 = scaling * np.concatenate(self.out1)
                out1 = time_window(out1, t_min, t_max)
                # print(f'Average V: {np.mean(out1)*1000:5.2f} mV')
                if t_min is None and not analogue_filter:
                    pt.plot(out1, label=label)
                elif analogue_filter:
                    ta, outa = _filter.get_awg_output(t, out1)
                    pt.plot(ta, outa, label=label)
                else:
                    pt.plot(t, out1, label=label)
        for i, m_list in enumerate(self.marker_out):
            if len(m_list) == 0:
                continue
            l = [[0, 0]]
            l += m_list
            l.append([t_end, 0])
            line = np.array(l).T
            label = plot_label + f'-M{i}'
            pt.plot(line[0], line[1], ':', label=label)
        limits = {}
        if t_min is not None:
            limits['left'] = t_min
        if t_max is not None:
            limits['right'] = t_max
        if limits:
            pt.xlim(**limits)

    def set_mock_data(self, bins, data: Iterable[Sequence[float]]):
        self.mock_data[bins] = iter(data)

    def get_acquisition_data(self):
        return (self.acq_count, self.acq_data, self.acq_thresholded)

    def get_acquisition_list(self):
        return self.acq_times


class AcqBuffer:
    def __init__(self):
        self.buffer = []
        self.write_ready = 0

    def add(self, time):
        b = self.buffer
        while len(b) and self.write_ready <= time:
            acq = b.pop(0)
            write_start = max(acq, self.write_ready)
            self.write_ready = write_start + 300
        overflow = len(b) > 7
        if not overflow:
            b.append(time)
        return not overflow
