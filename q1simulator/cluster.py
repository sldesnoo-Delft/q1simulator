import logging
from typing import Optional
from functools import partial

import qcodes as qc

from qblox_instruments import (
        SystemStatus, SystemState, SystemStatusSlotFlags,
        InstrumentClass, InstrumentType,
        )

from .q1simulator import Q1Module
from .triggers import TriggerDistributor
from .trigger_sorting import get_seq_trigger_info, sort_sequencers

logger = logging.getLogger(__name__)


class ClusterModule(qc.InstrumentChannel, Q1Module):
    def __init__(self, root_instrument, name, slot, n_sequencers=6, sim_type=None):
        super().__init__(root_instrument, name)
        self._slot = slot
        super().init_module(n_sequencers, sim_type)

    def present(self):
        return True

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

    def __init__(self, name, modules={}):
        super().__init__(name)

        # TODO return trigger count
        for par_name in self._cluster_parameters:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))
        for i in range(1,16):
            par_name = f'trigger{i}_monitor_count'
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

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

        # collect sequencers and sort on used triggers
        # TODO Refactor trigger distribution. Change into runtime distributor.
        seq_infos = []
        for module in modules:
            seq_numbers = [sequencer] if sequencer is not None else module.armed_seq
            for seq_number in seq_numbers:
                seq_infos.append(
                        get_seq_trigger_info(module, seq_number, module.sequencers[seq_number])
                        )

        trigger_dist = TriggerDistributor()
        seq_infos = sort_sequencers(seq_infos)
        for seq_info in seq_infos:
            seq = seq_info.module.sequencers[seq_info.seq_number]
            seq.set_trigger_events(trigger_dist.get_trigger_events())
            seq.run()
            trigger_dist.add_emitted_triggers(seq.get_acq_trigger_events())

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

