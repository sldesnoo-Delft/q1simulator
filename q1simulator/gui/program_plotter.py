from typing import Any

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from qtpy import QtWidgets, QtCore

from q1simulator import Q1Simulator
from q1simulator.channel_data import MarkerOutput, SampledOutput


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
        pw.getAxis('bottom').enableAutoSIPrefix(False)
        pw.getAxis('bottom').setLabel("time", "ns")
        pw.getAxis('left').setLabel("output", "V")
        pw.addLegend()
        self.proxy = pg.SignalProxy(pw.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self._time_label = pg.LabelItem("-- ns")
        self._time_label.setParentItem(pw.getPlotItem().vb)
        self._time_label.anchor(itemPos=(0.0, 0.0), parentPos=(0.0, 0.0), offset=(35, 10))

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self._plot_widget, 1)

        content.setLayout(layout)
        self.setCentralWidget(content)

    def mouseMoved(self, evt):
        viewbox = self._plot_widget.getPlotItem().vb
        pos = evt[0]  # using signal proxy turns original arguments into a tuple
        if viewbox.sceneBoundingRect().contains(pos):
            mousePoint = viewbox.mapSceneToView(pos)
            x_val = mousePoint.x()
            self._time_label.setText(f"{x_val:.0f} ns")

    def plot(self, x, y, style: str = "", label: str = None):
        color = color_list[self._n_plots % len(color_list)]
        if style == ":":
            pen = pg.mkPen(color, style=QtCore.Qt.DashLine)
        else:
            pen = color
        pw = self._plot_widget
        pw.plot(x, y, name=label, pen=pen)
        self._n_plots += 1

    def closeEvent(self, event):
        PlotWindow._windows.remove(self)


def plot_simulation(
        path: str,
        config: dict[str, Any],
        channels: list[str],
        min_time: int | None = None,
        max_time: int | None = None,
        analogue_filter: bool = False,
        max_core_cycles: int = 10_000_000,
        render_repetitions: bool = False,
        skip_wait_sync: bool = True,
        analogue_output_frequency: float = 4e9,
        plot_i_only: bool = True,
        trace: bool = False,
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
    sim.config('trace', trace)
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
        status = sim.get_sequencer_status(i)
        if status.warn_flags or status.err_flags:
            print(f"State {ch_name}: {status.status} - {status.state}, {[str(flag) for flag in status.info_flags]}")
            print(f"  warnings: {[str(flag) for flag in status.warn_flags]}, "
                  f"errors: {[str(flag) for flag in status.err_flags]}, log: {status.log}")

    # plot
    output_per_sequencer = True # TODO  @@@
    output = sim.get_output(
        t_min=min_time,
        t_max=max_time,
        analogue_filter=analogue_filter,
        output_frequency=analogue_output_frequency,
        output_per_sequencer=output_per_sequencer,
        )

    pw = PlotWindow()

    for name, data in output.items():
        if output_per_sequencer:
            is_marker = len(name) > 3 and name[-3:-1] == '-M'
        else:
            is_marker = name[0] == "M"
        linestyle = ":" if is_marker else "-"
        if isinstance(data, MarkerOutput):
            x, y = data.get_xy_lines()
            pw.plot(x, y, linestyle, label=name)
        elif isinstance(data, SampledOutput):
            t = data.get_time_data()
            if plot_i_only and name.endswith("-Q"):
                continue
            pw.plot(t, data.data, linestyle, label=name)

    acq_windows = sim.get_acquisition_windows()
    for name, data in acq_windows.items():
        n = len(data)
        t_list = [data[i][0] for i in range(n)]
        i_list = [data[i][1] for i in range(n)]
        q_list = [data[i][2] for i in range(n)]
        t = _concat_with_NaN(t_list)
        i = _concat_with_NaN(i_list)
        q = _concat_with_NaN(q_list)
        if not plot_i_only and not np.array_equal(i, q, equal_nan=True):
            i_label = name+"-I"
            q_label = name+"-Q"
        else:
            i_label = name
            q = None
        pw.plot(t, i, ":", label="ACQ:"+i_label)
        if q is not None:
            pw.plot(t, q, ":", label="ACQ:"+q_label)

    sim.close()

    return pw


def _concat_with_NaN(x: list[NDArray]) -> NDArray:
    nan_array = np.array([np.nan])
    t = []
    for i, d in enumerate(x):
        if i > 0:
            t.append(nan_array)
        t.append(d)
    return np.concatenate(t)
