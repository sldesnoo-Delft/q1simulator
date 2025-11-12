import logging
from functools import partial

import matplotlib.pyplot as pt
import qcodes as qc

from qblox_instruments import (
        InstrumentClass, InstrumentType,
        SystemStatuses, SystemStatus, SystemStatusSlotFlags,
        )

from .qblox_version import check_qblox_instrument_version
from .q1simulator import Q1Module, run_sequencers


logger = logging.getLogger(__name__)


class ClusterModule(qc.InstrumentChannel, Q1Module):
    def __init__(self, root_instrument, name, slot, n_sequencers=6, sim_type=None):
        super().__init__(root_instrument, name)
        self._slot = slot
        super().init_module(n_sequencers, sim_type)

    def present(self):
        return True

    @property
    def slot_idx(self):
        return self._slot


class EmptySlot(qc.InstrumentChannel):
    def __init__(self, root_instrument, name):
        super().__init__(root_instrument, name)

    def present(self):
        return False


class Cluster(qc.Instrument):
    _log_only_params = [
        'led_brightness',
        'reference_source',
        'ext_trigger_input_delay',
        'ext_trigger_input_trigger_en',
        'ext_trigger_input_trigger_address',
        'trigger_monitor_latest',
        ]

    def __init__(self, name, modules={}):
        check_qblox_instrument_version()
        super().__init__(name)

        # TODO return trigger count
        for i in range(1, 16):
            par_name = f'trigger{i}_monitor_count'
            self.add_parameter(par_name, set_cmd=partial(self._log_set, par_name))

        for par_name in self._log_only_params:
            self.add_parameter(par_name, set_cmd=partial(self._log_set, par_name))

        self._modules = {}
        for slot in range(1, 21):
            name = f'module{slot}'
            if slot in modules:
                module = ClusterModule(self, name, slot, sim_type=modules[slot])
            else:
                module = EmptySlot(self, name)
            self.add_submodule(name, module)
            self._modules[slot] = module

    def get_idn(self):
        return dict(vendor='Q1Simulator', model='Cluster', serial='', firmware='')

    @property
    def instrument_class(self):
        return InstrumentClass.CLUSTER

    @property
    def instrument_type(self):
        return InstrumentType.MM

    @property
    def modules(self):
        return list(self.submodules.values())

    def reset(self):
        self.invalidate_cache()
        for module in self.get_connected_modules().values():
            module.reset()

    def get_num_system_error(self):
        return 0

    def get_system_error(self):
        return '0,"No error"'

    def get_system_status(self):
        return SystemStatus(
            SystemStatuses.OKAY,
            [],
            SystemStatusSlotFlags({}))

    def get_connected_modules(self, filter_fn=None):
        result = {}
        for slot, module in self._modules.items():
            if module.present():
                result[slot] = module
        return result

    def _check_module_present(self, slot):
        if not self._modules[slot].present():
            raise Exception(f'No module in slot {slot}')

    def get_sequencer_status(self, slot, seq_nr, timeout=0):
        return self._modules[slot].get_sequencer_status(seq_nr, timeout)

    def get_acquisition_status(self, slot, seq_nr, timeout=0):
        return self.self._modules[slot].get_acquisition_status(seq_nr, timeout)

    def arm_sequencer(self, slot: int | None = None, sequencer: int | None = None) -> None:
        if slot is not None:
            self._check_module_present(slot)
            self._modules[slot].arm_sequencer(sequencer)
        else:
            for module in self.get_connected_modules().values():
                module.arm_sequencer(sequencer)

    def start_sequencer(self, slot: int | None = None, sequencer: int | None = None) -> None:
        if slot is not None:
            self._check_module_present(slot)
            modules = [self._modules[slot]]
        else:
            modules = self.get_connected_modules().values()

        # Get list of armed sequencers
        # pass to sequence executor
        sequencers = []
        for module in modules:
            seq_numbers = [sequencer] if sequencer is not None else module.armed_sequencers
            for seq_number in seq_numbers:
                sequencers.append(module.sequencers[seq_number])

        run_sequencers(sequencers)

    def stop_sequencer(self, slot: int | None = None, sequencer: int | None = None) -> None:
        if slot is not None:
            self._check_module_present(slot)
            self._modules[slot].stop_sequencer(sequencer)
        else:
            for module in self.get_connected_modules().values():
                module.stop_sequencer(sequencer)

    def get_waveforms(self, slot: int, sequencer: int, *, as_numpy: bool = False):
        return self._modules[slot].get_waveforms(sequencer, as_numpy=as_numpy)

    def get_weights(self, slot: int, sequencer: int, *, as_numpy: bool = False):
        return self._modules[slot].get_weights(sequencer, as_numpy=as_numpy)

    def store_scope_acquisition(self, slot: int, sequencer: int, name: str):
        self._modules[slot].store_scope_acquisition(sequencer, name)

    def get_acquisitions(self, slot: int, sequencer: int, *, as_numpy: bool = False) -> dict:
        return self._modules[slot].get_acquisitions(sequencer, as_numpy=as_numpy)

    def delete_acquisition_data(self, slot: int, sequencer: int, name: str = '', all=False):
        return self._modules[slot].delete_acquisition_data(slot, name=name, all=all)

    def connect_sequencer(self, slot: int, sequencer: int, *connections: str) -> None:
        self._modules[slot].connect_sequencer(sequencer, *connections)

    def disconnect_inputs(self, slot: int) -> None:
        self._modules[slot].disconnect_inputs()

    def disconnect_outputs(self, slot: int) -> None:
        self._modules[slot].disconnect_outputs()

    def _log_set(self, name, value):
        logger.info(f"{self.name}: {name}={value}")

    # ---- Simulator specific methods ----

    def config(self, name, value):
        for module in self.get_connected_modules().values():
            module.config(name, value)

    def get_simulation_end_time(self):
        return max(module.get_simulation_end_time() for module in self.get_connected_modules().values())

    def plot(self,
             t_min: float | None = None,
             t_max: float | None = None,
             channels: list[str] | list[int] | None = None,
             modules: list[int] | None = None,
             create_figure: bool | str = True,
             analogue_filter: bool = False,
             analogue_output_frequency: float = 4e9,
             output_per_sequencer: bool = True,
             **kwargs):
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
            output_per_sequencer:
                if True: plot data for individual sequencers.
                if False: plot data for physical front panel output.
        """
        if create_figure is True:
            pt.figure()
            pt.title('Cluster')
            pt.grid(True)
            pt.xlabel('[ns]')
            pt.ylabel('[V]')
        for slot, module in self.get_connected_modules().items():
            if modules is not None and slot not in modules:
                # skip module
                continue
            if create_figure == "module":
                pt.figure()
                pt.title(module.label)
                pt.grid(True)
                pt.xlabel('[ns]')
                pt.ylabel('[V]')
            module.plot(t_min=t_min, t_max=t_max, channels=channels, analogue_filter=analogue_filter,
                        analogue_output_frequency=analogue_output_frequency,
                        output_per_sequencer=output_per_sequencer)
            pt.legend()
        pt.show()
