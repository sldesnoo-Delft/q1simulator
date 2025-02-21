from numbers import Number
from packaging.version import Version

import pyqtgraph as pg
from qtpy import QtWidgets

from q1simulator import Q1Simulator
from q1simulator.qblox_version import qblox_version


# default colors cycle: see matplotlib CN colors.
color_cycle = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]
color_list = [pg.mkColor(cl) for cl in color_cycle]

_sim_counter = 0


class PlotWindow(QtWidgets.QMainWindow):

    _windows = []

    def __init__(
        self,
    ):
        super().__init__()
        self._n_plots = 0
        self.setWindowTitle("Simulated output")
        self.resize(800, 600)
        self.create_ui()
        self.show()

        # add to static list to avoid garbage collection.
        PlotWindow._windows.append(self)

    def create_ui(self):
        content = QtWidgets.QWidget()

        pw = pg.PlotWidget()
        self._plot_widget = pw
        pw.getAxis('bottom').setLabel("time", "ns")
        pw.getAxis('left').setLabel("output", "V")
        pw.addLegend()

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._plot_widget, 1)

        content.setLayout(layout)
        self.setCentralWidget(content)

    def plot(self, x, y, style: str = "", label: str = None):
        color = color_list[self._n_plots % len(color_list)]
        pw = self._plot_widget
        pw.plot(x, y, name=label, pen=color)
        self._n_plots += 1

    def closeEvent(self, event):
        PlotWindow._windows.remove(self)


def plot_simulation(
        path: str,
        config: dict[str, any],
        channels: list[str],
        min_time: int | None = None,
        max_time: int | None = None,
        analogue_filter: bool = False,
        max_core_cycles: int = 10_000_000,
        render_repetitions: bool = False,
        skip_wait_sync: bool = True,
        ):
    global _sim_counter

    max_render_time = max_time if max_time else 2e6

    # create simulator
    sim = Q1Simulator(
        f"Q1Viewer_{_sim_counter}",
        n_sequencers=len(channels),
        sim_type="Viewer")

    sim.config("max_render_time", max_render_time)
    sim.config("max_core_cycles", max_core_cycles)
    sim.config('skip_loops', () if render_repetitions else ("_start", ))
    sim.config('skip_wait_sync', skip_wait_sync)
    # sim.config('trace', True)
    sim.ignore_triggers = True

    _sim_counter += 1

    # config channels and  load programs
    for i, ch_name in enumerate(channels):
        ch_conf = config[ch_name]

        sequencer = sim.sequencers[i]
        sequencer.label = ch_name
        f_nco = ch_conf["nco"]
        if f_nco is None:
            sequencer.mod_en_awg(False)
        else:
            sequencer.mod_en_awg(True)
            sequencer.nco_freq(f_nco)

        for ch in ch_conf["out_channels"]:
            out_path = ch % 2
            sequencer.parameters[f'connect_out{ch}'].set('I' if out_path == 0 else 'Q')

        sequencer.sequence(path + "/" + ch_conf["sequence"])
        sequencer.sync_en(True)

        sim.arm_sequencer(i)

    # run
    sim.start_sequencer()

    for i, ch_name in enumerate(channels):
        if qblox_version < Version("0.12"):
            state = sim.get_sequencer_state(i)
            print(f"State {ch_name}: {state.status}, {[str(flag) for flag in state.flags]}")
        else:
            status = sim.get_sequencer_status(i)
            print(f"State {ch_name}: {status.status} - {status.state}, {[str(flag) for flag in status.info_flags]}")
            if status.warn_flags or status.err_flags:
                print(f"  warnings: {[str(flag) for flag in status.warn_flags]}, "
                      f"errors: {[str(flag) for flag in status.err_flags]}, log: {status.log}")

    # plot
    output = sim.get_output(
        t_min=min_time,
        t_max=max_time,
        analogue_filter=analogue_filter,
        )

    pw = PlotWindow()

    for name, data in output.items():
        t, out = data
        if isinstance(t, Number):
            pw.plot(out, label=name)
        elif len(name) > 4 and name[-3:-1] == '-M':
            # marker output
            pw.plot(t, out, ":", label=name)
        else:
            pw.plot(t, out, label=name)

    # sim.print_acquisitions()

    sim.close()

    return pw
