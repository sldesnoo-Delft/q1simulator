import logging
from typing import Optional
from functools import partial

import qcodes as qc

from qblox_instruments import (
        SystemStatus, SystemState, SystemStatusSlotFlags,
        InstrumentClass, InstrumentType,
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
        for i in range(1,16):
            par_name = f'trigger{i}_monitor_count'
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        for par_name in self._log_only_params:
            self.add_parameter(par_name,
                               set_cmd=partial(self._log_set, par_name))

        self._modules = {}
        for slot in range(1,21):
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

    def get_system_state(self):
        return SystemState(
            SystemStatus.OKAY,
            [],
            SystemStatusSlotFlags({}))

    def _check_module_present(self, slot):
        if not self._modules[slot].present():
            raise Exception(f'No module in slot {slot}')

    def _set(self, name, value):
        logger.info(f'{self.name}:{name}={value}')

    def arm_sequencer(self, slot: Optional[int] = None, sequencer: Optional[int] = None) -> None:
        if slot is not None:
            self._check_module_present(slot)
            self._modules[slot].arm_sequencer(sequencer)
        else:
            for module in self._modules.values():
                if module.present():
                    module.arm_sequencer(sequencer)

    def start_sequencer(self, slot: Optional[int] = None, sequencer: Optional[int] = None) -> None:
        if slot is not None:
            self._check_module_present(slot)
            modules = [self._modules[slot]]
        else:
            modules = [
                    module for module in self._modules.values()
                    if module.present()
                    ]

        # Get list of armed sequencers
        # pass to sequence executor
        sequencers = []
        for module in modules:
            seq_numbers = [sequencer] if sequencer is not None else module.armed_seq
            for seq_number in seq_numbers:
                sequencers.append(module.sequencers[seq_number])

        run_sequencers(sequencers)

    def stop_sequencer(self, slot: Optional[int] = None, sequencer: Optional[int] = None) -> None:
        if slot is not None:
            self._check_module_present(slot)
            self._modules[slot].stop_sequencer(sequencer)
        else:
            for module in self._modules.values():
                if module.present():
                    module.stop_sequencer(sequencer)

    @property
    def modules(self):
        return list(self.submodules.values())

    def reset(self):
        pass

    def _log_set(self, name, value):
        logger.info(f'{self.name}: {name}={value}')

    def config(self, name, value):
        for module in self._modules.values():
            if module.present():
                module.config(name, value)
