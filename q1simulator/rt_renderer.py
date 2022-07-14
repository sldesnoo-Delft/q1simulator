import logging
from copy import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as pt

@dataclass
class Settings:
    marker : int = 0
    awg_offs0 : int = 0
    awg_offs1 : int = 0
    awg_gain0 : int = 0
    awg_gain1 : int = 0
    phase: Optional[float] = None
    phase_shift : float = 0

def _phase2float(arg0, arg1, arg2):
    phase = ((arg2/6250 + arg1)/400 + arg0)/400
#    print(f'Phase {phase:6.3f}')
    return phase

def float2int16array(value):
    return np.array(value*2**15, dtype=np.int32)

def _i16(value):
    return np.int16(value)

class Renderer:

    def __init__(self, name):
        self.name = name
        self.max_render_time = 2_000_000
        self.wavedict = {}
        self.acq_weights = {}
        self.acq_bins = {}
        self.path_out_enabled = [set(), set()]
        self.nco_frequency = 0.0
        self.mod_en_awg = False
        self.nco_phase_offset = 0.0
        self.mixer_gain_ratio = 1.0
        self.mixer_phase_offset_degree = 0.0
        self.reset()

    def reset(self):
        self.settings = Settings()
        self.next_settings = Settings()
        self.time = 0
        self.wave_start = 0
        self.waves_end = (0,0)
        self.waves = (None, None)
        self.out0 = []
        self.out1 = []
        self.acq_count = {
                i:[0]*num_bins for i,num_bins in self.acq_bins.items()
                }
        self.acq_data = {
                i:[float('nan')]*num_bins for i,num_bins in self.acq_bins.items()
                }
        self.acq_times = {i:[] for i in self.acq_bins}
        self.acq_buffer = AcqBuffer()
        self.errors = set()

    def path_enable(self, path, out, enable):
        if enable:
            self.path_out_enabled[path].add(out)
        else:
            self.path_out_enabled[path].discard(out)

    def set_waveforms(self, wavedict):
        self.wavedict = wavedict

    def set_weights(self, weightsdict):
        self.acq_weights = weightsdict

    def set_acquisition_bins(self, acq_bins):
        self.acq_bins = acq_bins

    def set_mrk(self, value):
        self.next_settings.marker = value

    def reset_ph(self):
        self.next_settings.phase = 0.0

    def set_ph(self, arg0, arg1, arg2):
        self.next_settings.phase = _phase2float(arg0, arg1, arg2)

    def set_ph_delta(self, arg0, arg1, arg2):
        self.next_settings.phase_shift = _phase2float(arg0, arg1, arg2)

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
        if wave0 not in self.wavedict:
            self._error('AWG WAVE PLAYBACK INDEX INVALID PATH 0')
        elif wave1 not in self.wavedict:
            self._error('AWG WAVE PLAYBACK INDEX INVALID PATH 1')
        else:
            self.wave_start = self.time
            self.waves = (float2int16array(self.wavedict[wave0]),
                          float2int16array(self.wavedict[wave1]))
            self.waves_end = (self.time + len(self.waves[0]),
                              self.time + len(self.waves[1]))
        self._render(wait_after)

    def acquire(self, bins, bin_index, wait_after):
        self._update_settings()
        self._add_acquisition(bins, bin_index)
        self._render(wait_after)

    def acquire_weighed(self, bins, bin_index, weight0, weight1, wait_after):
        self._update_settings()
        if weight0 not in self.acq_weights:
            self._error('ACQ WEIGHT PLAYBACK INDEX INVALID PATH 0')
        elif weight1 not in self.acq_weights:
            self._error('ACQ WEIGHT PLAYBACK INDEX INVALID PATH 1')
        else:
            self._add_acquisition(bins, bin_index)
        self._render(wait_after)

    def wait(self, time):
        self._render(time)

    def wait_sync(self, wait_after):
        self._render(wait_after)

    def _error(self, msg):
        logging.error(f'{self.name}: {msg}')
        self.errors.add(msg)

    def _update_settings(self):
        new = self.next_settings
        if new.phase is not None:
            phase = (new.phase - self.time * self.nco_frequency * 1e-9) % 1
            self.nco_phase_offset = phase
        self.nco_phase_offset += new.phase_shift
        self.settings = self.next_settings
        self.settings.phase = None
        self.next_settings = copy(self.settings)
        self.next_settings.phase = None
        self.next_settings.phase_shift = 0.0

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
            phase = (2*np.pi*self.nco_phase_offset + 2*np.pi*self.nco_frequency * 1e-9 * t)
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
            if self.acq_count[bins][bin_index] == 0:
                self.acq_data[bins][bin_index] = t
            else:
                self.acq_data[bins][bin_index] += t
            self.acq_count[bins][bin_index] += 1

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

