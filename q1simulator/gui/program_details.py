import json
from qtpy import QtWidgets, QtCore

from .program_plotter import plot_simulation


class ProgramDetailsWidget(QtWidgets.QWidget):
    def __init__(self,
                 parent: QtWidgets.QWidget,
                 ):
        super().__init__(parent)
        self._prog_config = None
        self._path = None
        self.trace_simulation = False

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self._table = self.create_table()
        layout.addWidget(self._table, 1)
        layout.addLayout(self.create_controls())

        self._plot_btn.clicked.connect(self.plot)

    def create_table(self):
        table = QtWidgets.QTreeWidget()
        table.setMinimumHeight(400)
        table.setMinimumWidth(540)
        table.setColumnCount(7)
        table.setColumnWidth(0, 100)
        table.setColumnWidth(1, 100)
        table.setColumnWidth(2, 50)
        table.setColumnWidth(3, 50)
        table.setColumnWidth(4, 50)
        table.setColumnWidth(5, 90)
        table.setColumnWidth(6, 80)
        table.setHeaderLabels([
            "Channel",
            "Module",
            "Paths",
            "Out",
            "In",
            "NCO",
            "Duration",
            ])
        table.setRootIsDecorated(False)
        return table

    def create_controls(self):
        min_time = QtWidgets.QSpinBox()
        min_time.setRange(0, 100_000_000)
        min_time.setSingleStep(1000)
        self._min_time_spin = min_time
        max_time = QtWidgets.QSpinBox()
        max_time.setSpecialValueText("Duration")
        max_time.setRange(0, 100_000_000)
        max_time.setSingleStep(1000)
        self._max_time_spin = max_time

        self._iq_plot = QtWidgets.QComboBox()
        self._iq_plot.addItems(["I-only", "I+Q"])

        self._analogue_cb = QtWidgets.QCheckBox("Analogue filter")
        self._plot_btn = QtWidgets.QPushButton("Plot")
        self._plot_btn.setMinimumWidth(80)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("t min."))
        layout.addWidget(min_time)
        layout.addWidget(QtWidgets.QLabel("t max."))
        layout.addWidget(max_time)
        layout.addWidget(QtWidgets.QLabel("IQ"))
        layout.addWidget(self._iq_plot)
        layout.addWidget(self._analogue_cb)
        layout.addItem(QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))
        layout.addWidget(self._plot_btn)

        return layout

    def show_details(self, path):
        table = self._table
        table.clear()
        self._prog_config = None
        try:
            with open(path + "/q1program.json", "r") as fp:
                data = json.load(fp)
            self._path = path
            self._prog_config = data
            duration = -1
            rows = []
            for ch_name, settings in data.items():
                f_nco = settings["nco"]
                seq_duration = settings.get("duration")
                if seq_duration is not None:
                    duration = max(seq_duration, duration)
                    duration_text = f"{seq_duration} ns"
                else:
                    duration_text = "--"
                q1asm = settings.get("sequence")
                if not q1asm:
                    duration_text = "no sequence"

                item = QtWidgets.QTreeWidgetItem(table)
                item.setText(0, ch_name)
                item.setData(0, QtCore.Qt.CheckStateRole, QtCore.Qt.Checked if q1asm else QtCore.Qt.Unchecked)

                item.setText(1, settings["module"])
                item.setText(2, str(settings["paths"]))
                item.setText(3, str(settings["out_channels"]))
                item.setText(4, str(settings.get("in_channels", "")))
                item.setText(5, f"{f_nco/1e6} MHz" if f_nco is not None else "")
                item.setText(6, duration_text)
                rows.append(item)
            table.insertTopLevelItems(0, rows)

            # Show duration

        except FileNotFoundError:
            print("'program_config.json' not found")

    def plot(self):
        if not self._prog_config:
            print("no program")
            return
        table = self._table
        channels = []
        for i in range(table.topLevelItemCount()):
            item = table.topLevelItem(i)
            if item.checkState(0) == QtCore.Qt.Checked:
                channels.append(item.text(0))

        min_time = self._min_time_spin.value()
        max_time = self._max_time_spin.value()
        if max_time == 0:
            max_time = None
        plot_i_only = self._iq_plot.currentText() == "I-only"

        plot_simulation(
            self._path,
            self._prog_config,
            channels,
            min_time=min_time,
            max_time=max_time,
            analogue_filter=self._analogue_cb.isChecked(),
            analogue_output_frequency=4e9,
            plot_i_only=plot_i_only,
            trace=self.trace_simulation,
            )
