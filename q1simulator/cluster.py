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
    _cluster_parameters = [
        'trigger_monitor_latest',
        ]
    _log_only_params = [
        'led_brightness',
        ]

    def __init__(self, name, modules={}):
        check_qblox_instrument_version()
        super().__init__(name)

        # TODO return trigger count
        for par_name in self._cluster_parameters:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))
        for i in range(1, 16):
            par_name = f'trigger{i}_monitor_count'
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        for par_name in self._log_only_params:
            self.add_parameter(par_name,
                               set_cmd=partial(self._log_set, par_name))

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

    def _set(self, name, value):
        logger.info(f'{self.name}:{name}={value}')

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
            seq_numbers = [sequencer] if sequencer is not None else module.armed_seq
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

    @property
    def modules(self):
        return list(self.submodules.values())

    def reset(self):
        self.invalidate_cache()
        for module in self.get_connected_modules().values():
            module.reset()

    def _log_set(self, name, value):
        logger.info(f'{self.name}: {name}={value}')

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
                        analogue_output_frequency=analogue_output_frequency)
            pt.legend()
        pt.show()
