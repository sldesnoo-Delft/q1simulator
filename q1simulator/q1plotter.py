import logging
import os
from packaging.version import Version

from qcodes import Instrument
from qblox_instruments import Cluster
from .cluster import Cluster as SimCluster
from .qblox_version import qblox_version


logger = logging.getLogger(__name__)


class Q1Plotter:
    def __init__(self,
                 cluster: Cluster,
                 max_render_time=2e6,
                 max_core_cycles=1e7,
                 skip_loops=("_start", ),
                 skip_wait_sync=True,
                 acq_trigger_value=0,
                 ):

        self._cluster = cluster
        self._simulator = SimCluster(self._get_name(), self._get_module_types())
        self.max_render_time = max_render_time
        self.max_core_cycles = max_core_cycles
        self.skip_loops = skip_loops
        self.skip_wait_sync = skip_wait_sync
        self.acq_trigger_value = acq_trigger_value

        self.load()

    @property
    def max_render_time(self):
        """Maximum time the output is calculated."""
        return self._max_render_time

    @max_render_time.setter
    def max_render_time(self, value):
        self._max_render_time = value
        self._simulator.config('max_render_time', self._max_render_time)

    @property
    def max_core_cycles(self):
        """Maximum number of FPGA clock simulated per sequencer."""
        return self._max_core_cycles

    @max_core_cycles.setter
    def max_core_cycles(self, value):
        self._max_core_cycles = value
        self._simulator.config("max_core_cycles", self._max_core_cycles)

    @property
    def skip_loops(self) -> tuple[str, ...]:
        """Labels of loops to skip.
        Setting this to ("_start", ) will skip the repetitions of Q1Pulse sequences.
        """
        return self._skip_loops

    @skip_loops.setter
    def skip_loops(self, value):
        if value is None:
            value = ()
        self._skip_loops = value
        self._simulator.config("skip_loops", self._skip_loops)

    @property
    def skip_wait_sync(self):
        """If True the time after `wait_sync` is not added to the output."""
        return self._skip_wait_sync

    @skip_wait_sync.setter
    def skip_wait_sync(self, value):
        self._skip_wait_sync = value
        self._simulator.config('skip_wait_sync', self._skip_wait_sync)

    @property
    def acq_trigger_value(self) -> int | None:
        """The value used for every acquisition trigger in the sequence.
        Allowed values: None, 0, 1.

        If None the acquired data is used to determine the trigger value.
        """
        return self._acq_trigger_value

    @acq_trigger_value.setter
    def acq_trigger_value(self, value):
        self._acq_trigger_value = value
        self._simulator.config('acq_trigger_value', self._acq_trigger_value)

    @staticmethod
    def _get_name():
        counter = 1
        name = f'Q1Plotter_{counter}'
        while Instrument.exist(name):
            counter += 1
            name = f'Q1Plotter_{counter}'
        return name

    def _get_module_types(self):
        modules = {}
        for slot, module in enumerate(self._cluster.modules, 1):
            if module.present():
                if module.is_qcm_type:
                    sim_type = "QCM"
                elif module.is_qrm_type:
                    sim_type = "QRM"
                else:
                    logger.warning(f"unknown module type {module.module_type} in slot {slot}")
                    continue
                if module.is_rf_type:
                    sim_type += "_RF"
                modules[slot] = sim_type
        return modules

    def load(self):
        """Loads program and settings from the cluster and generates the output."""
        for slot, module in enumerate(self._simulator.modules, 1):
            if module.present():
                cluster_mod = self._cluster.modules[slot-1]
                module.label = cluster_mod.label
                for iseq in range(6):
                    cluster_seq = cluster_mod.sequencers[iseq]
                    sim_seq = module.sequencers[iseq]
                    enabled = self._copy_settings(cluster_seq, sim_seq, module.is_qcm_type, module.is_rf_type)
                    if enabled:
                        print(f"Generating output for slot {slot}, sequencer {iseq}: {sim_seq.label}")
                        module.arm_sequencer(iseq)

        self._simulator.start_sequencer()

    def regenerate(self):
        """Runs the program again to generate the output."""
        self._simulator.start_sequencer()

    def plot(self,
             t_min: float | None = None,
             t_max: float | None = None,
             channels: list[str] | list[int] | None = None,
             modules: list[int] | None = None,
             create_figure: bool | str = True,
             analogue_filter: bool = False,
             analogue_output_frequency: float = 4e9,
             ):
        """Plots the simulated output of the cluster.

        Args:
            t_min: minimum time in the plot.
            t_max: maximum time in the plot.
            channels: If not None specifies the channels to plot by name or sequencer number.
            modules: If not None specifies the modules to plot by slot number.
            create_figure:
                If True create a new figure.
                If False only pyplot.plot() is called without creating figure or setting axis labels.
                If "modules" creates a new figure per module.
            analogue_filter: plot result after applying (estimated) analog filter.
            analogue_output_frequency: sample rate of analogue output
        """
        end_time = self._simulator.get_simulation_end_time()
        if t_max is not None and t_max > self.max_render_time:
            print(f"The output has only been computed till t={self.max_render_time}. "
                  "Set `max_render_time` and call `regenerate()` to get more data.")
        if t_min is not None and t_min > end_time:
            print(f"The simulation ended at t={end_time}")

        self._simulator.plot(
            t_min=t_min,
            t_max=t_max,
            channels=channels,
            modules=modules,
            create_figure=create_figure,
            analogue_filter=analogue_filter,
            analogue_output_frequency=analogue_output_frequency,
            )

    @staticmethod
    def _copy_settings(cluster_seq, sim_seq, is_qcm, is_rf):
        sim_seq.sync_en(cluster_seq.sync_en.cache())
        if not sim_seq.sync_en():
            return False
        sequence = cluster_seq.sequence.cache()
        if not isinstance(sequence, dict):
            if not os.path.exists(sequence):
                raise Exception(f"Cannot load sequence {sequence}")
        sim_seq.sequence(sequence)
        sim_seq.label = cluster_seq.label

        param_names = [
                "mod_en_awg",
                "nco_freq",
                'marker_ovr_en',
                'marker_ovr_value',
                'nco_phase_offs',
                'gain_awg_path0',
                'gain_awg_path1',
                'offset_awg_path0',
                'offset_awg_path1',
                ]

        for i in range(1, 16):
            param_names += [
                f'trigger{i}_count_threshold',
                f'trigger{i}_threshold_invert',
                ]

        if qblox_version >= Version('0.11'):
            n_out_ch = 4 if is_qcm else 2
            if is_rf:
                n_out_ch //= 2
            for ch in range(n_out_ch):
                param_names += [f"connect_out{ch}"]
        else:
            param_names += [
                'channel_map_path0_out0_en',
                'channel_map_path1_out1_en',
                ]
            if is_qcm:
                param_names += [
                    'channel_map_path0_out2_en',
                    'channel_map_path1_out3_en',
                    ]

        if not is_qcm:
            param_names += [
                "demod_en_acq",
                "integration_length_acq",
                'thresholded_acq_rotation',
                'thresholded_acq_threshold',
                'thresholded_acq_trigger_en',
                'thresholded_acq_trigger_address',
                'thresholded_acq_trigger_invert',
                ]

        for param_name in param_names:
            value = cluster_seq.parameters[param_name].cache()
            if value is not None:
                sim_seq.parameters[param_name].set(value)

        return True
