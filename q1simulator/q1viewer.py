from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib.pyplot as pt

from .q1simulator import Q1Simulator


@dataclass
class PlotDef:
    filename: str
    sequencer_name: Optional[str] = None
    out: List[int] = field(default_factory=lambda:[0,1])
    lo_frequency: float = None


def plot_q1asm_file(filename, out=[0,1], lo_frequency=None,
                    max_render_time=2e6,
                    max_core_cycles=1e7):
    plot = PlotDef(filename, out=out, lo_frequency=lo_frequency)
    plot_q1asm_files([plot],
                     max_render_time=max_render_time,
                     max_core_cycles=max_core_cycles)


def plot_q1asm_files(plot_defs,
                     max_render_time=2e6,
                     max_core_cycles=1e7):

    sim = Q1Simulator('sim', n_sequencers=len(plot_defs))
    for i,plot in enumerate(plot_defs):
        prefix = f'sequencer{i}_'

        if plot.sequencer_name:
            sim.set(prefix + 'name', plot.sequencer_name)
        sim.set(prefix + 'max_render_time', max_render_time)
        sim.set(prefix + 'max_core_cycles', max_core_cycles)

        if plot.lo_frequency is None:
            sim.set(prefix + 'mod_en_awg', False)
        else:
            sim.set(prefix + 'mod_en_awg', True)
            sim.set(prefix + 'nco_freq', plot.lo_frequency)

        for ch in plot.out:
            path = ch % 2
            sim.set(prefix + f'channel_map_path{path}_out{ch}_en', True)

        sim.set(prefix + 'waveforms_and_program',
                plot.filename)

        sim.arm_sequencer(i)

    sim.start_sequencer()

    for i,plot in enumerate(plot_defs):
        name = plot.sequencer_name if plot.sequencer_name else f'seq{i}'
        print(f'State {name}: {sim.get_sequencer_state(i)}')

    pt.figure()
    sim.plot()
    pt.legend()

    sim.print_acquisitions()

    sim.close()