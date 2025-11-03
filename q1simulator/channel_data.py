from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SampledOutput:
    t_min: int
    t_max: int
    sample_rate: int  # GSa/s
    data: NDArray

    def get_time_data(self):
        """Time is [ns]"""
        n = (self.t_max-self.t_min)*self.sample_rate
        return np.linspace(self.t_min, self.t_max, n, endpoint=False)

    def __add__(self, rhs: SampledOutput):
        if self.t_min != rhs.t_min:
            raise Exception(f"t_min not equal for sample output: {self.t_min} <> {rhs.t_min}")
        if self.sample_rate != rhs.sample_rate:
            raise Exception(f"sample rates not equal for sample output: {self.sample_rate} <> {rhs.sample_rate}")

        t_max = max(self.t_max, rhs.t_max)
        if self.t_max < t_max:
            lhs_data = np.zeros(t_max * self.sample_rate)
            lhs_data[:self.t_max * self.sample_rate] = self.data
        else:
            lhs_data = self.data
        if rhs.t_max < t_max:
            rhs_data = np.zeros(t_max * self.sample_rate)
            rhs_data[:rhs.t_max * self.sample_rate] = rhs.data
        else:
            rhs_data = rhs.data

        return SampledOutput(self.t_min, t_max, self.sample_rate, lhs_data+rhs_data)


@dataclass
class MarkerOutput:
    t_min: int
    t_max: int
    points: list[tuple[int, int]]

    def get_xy_lines(self):
        line = np.array(self.points).T
        return (line[0], line[1])

    def to_samples(self):
        n = self.t_max - self.t_min
        data = np.zeros(n)
        ip = 0
        points = self.points
        while points[ip][0] < self.t_min:
            ip += 1
        it = points[ip][0]
        data[:it] = points[ip][1]
        ip += 1
        while ip+1 < len(points) and points[ip+1][0] <= self.t_max:
            it1 = points[ip][0]
            it2 = points[ip+1][0]
            data[it1:it2] = points[ip][1]
            ip += 2
        return SampledOutput(self.t_min, self.t_max, 1, data)

    def __add__(self, rhs: MarkerOutput | SampledOutput):
        if isinstance(rhs, MarkerOutput):
            rhs = rhs.to_samples()
        return self.to_samples() + rhs

    def __radd__(self, lhs: MarkerOutput | SampledOutput):
        return self + lhs
