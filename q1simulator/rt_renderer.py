import logging
from copy import copy
from dataclasses import dataclass
from typing import Optional, Sequence, Iterable, Union
from numbers import Number
from collections.abc import Sequence as AbcSequence

import numpy as np
import matplotlib.pyplot as pt

MockDataEntry = Union[float, complex, Sequence[float]]

@dataclass
class Settings:
    marker : int = 0
    awg_offs0 : int = 0
    awg_offs1 : int = 0
    awg_gain0 : int = 0
    awg_gain1 : int = 0
    reset_phase : bool = False
    relative_phase: Optional[float] = None
    phase_shift : float = 0
    frequency : Optional[float] = None

def _phase2float(phase_uint32):
    # phase in rotations, i.e. unit = 2 pi rad
    phase = phase_uint32/1e9
    # print(f'Phase {phase:9.6f} rotations ({phase*360:8.3f} deg)')
    return phase

def _freq2Hz(freq_uint32):
    # freq in Hz
    # convert uint to int if value > 2**31
    if freq_uint32 > 2**31:
        freq_int32 = freq_uint32 - 2**32
    else:
        freq_int32 = freq_uint32
    freq = freq_int32/4
    # print(f'Frequency {freq/1e6:9.6f} MHz')
    return freq

def float2int16array(value):
    return np.array(value*2**15, dtype=np.int32)

def _i16(value):
    return np.int16(value)

class Renderer:

    def __init__(self, name):
        self.name = name
        self.max_render_time = 2_000_000
        self.wavedict_float = {}
        self.wavedict = {}
        self.acq_weights = {}
        self.acq_bins = {}
        self.path_out_enabled = [set(), set()]
        self.nco_frequency = 0.0
        self.mod_en_awg = False
        self.mixer_gain_ratio = 1.0
        self.mixer_phase_offset_degree = 0.0
        self.delete_acquisition_data_all()
        self.reset()
        self.trace_enabled = False

    def reset(self):
        self.settings = Settings()
        self.next_settings = Settings()
        self.time = 0
        # Qblox sequencer has 3 different phase registers
        self.nco_phase_offset = 0.0
        self.relative_phase = 0.0
        self.delta_phase = 0.0
        self.wave_start = 0
        self.waves_end = (0,0)
        self.waves = (None, None)
        self.out0 = []
        self.out1 = []
        self.acq_times = {i:[] for i in self.acq_bins}
        self.acq_buffer = AcqBuffer()
        self.mock_data = {}
        self.errors = set()

    def path_enable(self, path, out, enable):
        if enable:
            self.path_out_enabled[path].add(out)
        else:
            self.path_out_enabled[path].discard(out)

    def set_waveforms(self, wavedict):
        self.wavedict_float = wavedict
        self.wavedict = {
                key:float2int16array(value)
                for key,value in wavedict.items()
                }

    def set_weights(self, weightsdict):
        self.acq_weights = weightsdict

    def set_acquisition_bins(self, acq_bins):
        self.acq_bins = acq_bins
        self.delete_acquisition_data_all()

    def set_mrk(self, value):
        self.next_settings.marker = value

    def set_freq(self, freq):
        self.next_settings.frequency =_freq2Hz(freq)

    def reset_ph(self):
        self.next_settings.reset_phase = True

    def set_ph(self, phase):
        self.next_settings.phase = _phase2float(phase)

    def set_ph_delta(self, phase_delta):
        self.next_settings.phase_shift = _phase2float(phase_delta)

    def set_awg_gain(self, gain0, gain1):
        self.next_settings.awg_gain0 = gain0
        self.next_settings.awg_gain1 = gain1

    def set_awg_offs(self, offset0, offset1):
        self.next_settings.awg_offs0 = offset0
        self.next_settings.awg_offs1 = offset1

    def upd_param(self, wait_after):
        self._update_settings()
        self._render(wait_after)

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

    def acquire(self, bins, bin_index, wait_after):
        self._update_settings()
        if self.trace_enabled:
            self._trace(f'Acquire {bins} {bin_index}')
        self._add_acquisition(bins, bin_index)
        self._render(wait_after)

    def acquire_weighed(self, bins, bin_index, weight0, weight1, wait_after):
        self._update_settings()
        if self.trace_enabled:
            self._trace(f'AcquireWeighed {bins} {bin_index} {weight0} {weight1}')
        if weight0 not in self.acq_weights:
            self._error('ACQ WEIGHT PLAYBACK INDEX INVALID PATH 0')
        elif weight1 not in self.acq_weights:
            self._error('ACQ WEIGHT PLAYBACK INDEX INVALID PATH 1')
        else:
            self._add_acquisition(bins, bin_index)
        self._render(wait_after)

    def wait(self, time):
        if self.trace_enabled:
            self._trace(f'Wait {time}')
        self._render(time)

    def wait_sync(self, wait_after):
        self._render(wait_after)

    def _error(self, msg):
        logging.error(f'{self.name}: {msg}')
        self.errors.add(msg)

    def _update_settings(self):
        new = self.next_settings
        old = self.settings

        if self.trace_enabled:
            msg = []
            if new.awg_gain0 != old.awg_gain0 or new.awg_gain1 != old.awg_gain1:
                msg.append(f'awg_gain {new.awg_gain0},{new.awg_gain1}')
            if new.awg_offs0 != old.awg_offs0 or new.awg_offs1 != old.awg_offs1:
                msg.append(f'awg_offset {new.awg_offs0},{new.awg_offs1}')
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

        if new.phase is not None:
            phase = (new.phase - self.time * self.nco_frequency * 1e-9) % 1
            self.nco_phase_offset = phase
            self.next_settings.phase = None
        if new.frequency is not None:
            phase = (self.nco_phase_offset + self.time * self.nco_frequency * 1e-9) % 1
            new_phase_offset = phase - (self.time * new.frequency * 1e-9) % 1
            self.nco_phase_offset = new_phase_offset
            self.nco_frequency = new.frequency
            self.next_settings.frquency = None
        self.nco_phase_offset += new.phase_shift
        new.phase_shift = 0.0
        # copy offset and gain
        self.settings = copy(self.next_settings)

    def _render(self, time):
        if time & 0x0003:
            logging.error(f'{self.name}: wait time not aligned on '
                          f'4 ns boundary: {time} ns (offset={time&0x03} ns)')
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
            t_end = self.max_render_time

        t_render = t_end - t_start

        s = self.settings

        path0 = np.full(t_render, s.awg_offs0, dtype=np.int16)
        path1 = np.full(t_render, s.awg_offs1, dtype=np.int16)

        if self.waves_end[0] > t_start:
            end = min(self.waves_end[0], t_end)
            data = self.waves[0][t_start-self.wave_start:end-self.wave_start]
            path0[0:len(data)] += (int(_i16(s.awg_gain0)) * data // 2**15)
        if self.waves_end[1] > t_start:
            end = min(self.waves_end[1], t_end)
            data = self.waves[1][t_start-self.wave_start:end-self.wave_start]
            path1[0:len(data)] += (int(_i16(s.awg_gain1)) * data // 2**15)

        if self.mod_en_awg:
            t = np.arange(t_start, t_end)
            phase_offset = self.relative_phase + self.delta_phase
            if self.nco_frequency < 0:
                phase_offset = -phase_offset
            phase = 2*np.pi*(phase_offset + self.nco_phase_offset + self.nco_frequency * 1e-9 * t)
            lo = np.cos(phase) + 1j*np.sin(phase)
            data0 = lo.real*path0 - lo.imag*path1
            if self.mixer_phase_offset_degree != 0.0:
                phase_offset = self.mixer_phase_offset_degree/180*np.pi
                lo *= np.cos(phase_offset) + 1j*np.sin(phase_offset)
            data1 = lo.imag*path0 + lo.real*path1
            if self.mixer_gain_ratio > 1.0:
                data0 *= 1/self.mixer_gain_ratio
            if self.mixer_gain_ratio < 1.0:
                data1 *= self.mixer_gain_ratio
            data0 = data0.astype(np.int16)
            data1 = data1.astype(np.int16)
        else:
            data0 = path0
            data1 = path1

        if len(self.path_out_enabled[0]):
            self.out0.append(data0)
        if len(self.path_out_enabled[1]):
            self.out1.append(data1)

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
        for index in self.acq_bins:
            self.delete_acquisition_data(index)

    def delete_acquisition_data(self, index):
        num_bins = self.acq_bins[index]
        self.acq_count[index] = np.zeros(num_bins, dtype=int)
        self.acq_data[index] = np.full((num_bins, 2), np.nan)

    def _add_acquisition(self, bins, bin_index):
        t = self.time
        if bins not in self.acq_bins:
            self._error('ACQ INDEX INVALID')
            return
        self.acq_times[bins].append((t, bin_index))
        if bin_index >= self.acq_bins[bins]:
            self._error('ACQ BIN INDEX INVALID')
        elif not self.acq_buffer.add(t):
            self._error('ACQ BINNING FIFO ERROR')
        else:
            value = self._get_acq_data(bins, t)
            if self.acq_count[bins][bin_index] == 0:
                self.acq_data[bins][bin_index] = value
            else:
                self.acq_data[bins][bin_index] += value
            self.acq_count[bins][bin_index] += 1

    def _trace(self, msg):
        if self.trace_enabled:
            print(f'{self.time:-6} {msg}')

    def plot(self, v_max):
        scaling = v_max/2**15
        if self.time > self.max_render_time:
            max_ms = self.max_render_time / 1e6
            print(f'{self.name}: Rendering truncated at {max_ms:3.1f} ms. Total time: {self.time/1e6:4.1f} ms')
        if len(self.path_out_enabled[0]):
            out0 = scaling * np.concatenate(self.out0)
            pt.plot(out0, label=f'{self.name}.{self.path_out_enabled[0]}')
        if len(self.path_out_enabled[1]):
            out1 = scaling * np.concatenate(self.out1)
            pt.plot(out1, label=f'{self.name}.{self.path_out_enabled[1]}')

    def set_mock_data(self, bins, data: Iterable[Sequence[float]]):
        self.mock_data[bins] = iter(data)

    def get_acquisition_data(self):
        return (self.acq_count, self.acq_data)

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
            self.write_ready = write_start + 1040
        overflow = len(b) > 7
        if not overflow:
            b.append(time)
        return not overflow

