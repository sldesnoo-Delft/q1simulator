import os

import numpy as np
import xarray as xr


class AnalogueFilter:
    warned = False

    def __init__(self, model_name: str, output_frequency: float = 4e9):
        if not AnalogueFilter.warned:
            print(f"WARNING: Analogue output of simulated {model_name} can differ "
                  "significantly from real hardware!")
            AnalogueFilter.warned = True

        pulse_response_dir = os.path.dirname(__file__)
        if model_name == "QCM":
            fname = pulse_response_dir + "/Qblox_QCM_pulse_response.hdf5"
            amplitude_correction = 1.003
            self.min_vstep = 5.0/2**16  # 0.076 mV
        else:
            raise Exception(f"Uknown model_name {model_name}")

        pulse_response = xr.open_dataset(fname, engine='h5netcdf')
        t_response = pulse_response.coords['t'].data
        self.pulse_response = pulse_response['y'].data * amplitude_correction
        sr = round(1/(t_response[1]-t_response[0]))
        sub_sample = int(round(sr / (output_frequency*1e-9)))
        self.sr = sr // sub_sample
        self.pulse_response = self.pulse_response[::sub_sample]

        self.n_before = round(-t_response[0] * self.sr)
        self.n_after = len(self.pulse_response) - self.n_before - 1

    def quantize_amplitude(self, wave):
        return np.round(wave/self.min_vstep) * self.min_vstep

    def get_awg_output(self, t, samples, analogue_shift: float | None = 0.0):
        samples = self.quantize_amplitude(samples)
        if t[1] - t[0] != 1:
            raise Exception(f"Expecting input with 1 ns sample period. Got {t[1]-t[0]}")
        t = np.linspace(t[0], t[-1]+1, len(t)*self.sr, endpoint=False)
        d = np.zeros(len(samples)*self.sr)
        d[::self.sr] = samples

        d = np.convolve(d, self.pulse_response)
        return t+analogue_shift, d[self.n_before: -self.n_after]
