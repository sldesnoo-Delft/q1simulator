import sys
from dataclasses import dataclass, field
from packaging.version import Version
from typing import List, Optional

import matplotlib.pyplot as pt

from .q1simulator import Q1Simulator
from .qblox_version import qblox_version


@dataclass
class PlotDef:
    filename: str
    sequencer_name: Optional[str] = None
    out: List[int] = field(default_factory=lambda:[0,1])
    lo_frequency: Optional[float] = None


def plot_q1asm_file(filename,
                    out=[0,1],
                    lo_frequency=None,
                    max_render_time=2e6,
                    max_core_cycles=1e7,
                    render_repetitions=False,
                    skip_wait_sync=True,
                    t_min=None,
                    t_max=None,
                    ):
    plot = PlotDef(filename, out=out, lo_frequency=lo_frequency)
    plot_q1asm_files([plot],
                     max_render_time=max_render_time,
                     max_core_cycles=max_core_cycles,
                     render_repetitions=render_repetitions,
                     skip_wait_sync=skip_wait_sync,
                     t_min=t_min,
                     t_max=t_max,
                     )


def plot_q1asm_files(plot_defs,
                     max_render_time=2e6,
                     max_core_cycles=1e7,
                     render_repetitions=False,
                     skip_wait_sync=True,
                     t_min=None,
                     t_max=None,
                     ):
    sim = Q1Simulator('sim', n_sequencers=len(plot_defs), sim_type='Viewer')
    sim.config('max_render_time', max_render_time)
    sim.config('max_core_cycles', max_core_cycles)
    sim.config('skip_loops', () if render_repetitions else ("_start", ))
    sim.config('skip_wait_sync', skip_wait_sync)
    sim.ignore_triggers = True

    for i, plot in enumerate(plot_defs):
        if plot.sequencer_name:
            sim.sequencers[i].label = plot.sequencer_name

        sequencer = getattr(sim, f'sequencer{i}')
        if plot.lo_frequency is None:
            sequencer.set('mod_en_awg', False)
        else:
            sequencer.set('mod_en_awg', True)
            sequencer.set('nco_freq', plot.lo_frequency)

        for ch in plot.out:
            path = ch % 2
            if qblox_version < Version('0.11'):
                sequencer.set(f'channel_map_path{path}_out{ch}_en', True)
            else:
                sequencer.set(f'connect_out{ch}', 'I' if path == 0 else 'Q')

        sequencer.set('sequence', plot.filename)
        sequencer.sync_en(True)

        sim.arm_sequencer(i)

    sim.start_sequencer()

    for i,plot in enumerate(plot_defs):
        name = plot.sequencer_name if plot.sequencer_name else f'seq{i}'
        if qblox_version < Version("0.12"):
            print(f'State {name}: {sim.get_sequencer_state(i)}')
        else:
            print(f'State {name}: {sim.get_sequencer_status(i)}')

    sim.plot(t_min=t_min, t_max=t_max)
    sim.print_acquisitions()

    sim.close()


def main(argv):
    plot_q1asm_file(argv[0])
    pt.show()


if __name__ == "__main__":
   main(sys.argv[1:])
