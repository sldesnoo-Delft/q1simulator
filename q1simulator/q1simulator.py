import logging
from functools import partial

import matplotlib.pyplot as pt
import qcodes as qc

from .q1module import Q1Module
from .q1sequencer import Q1Sequencer
from .qblox_version import check_qblox_instrument_version
from .triggers import TriggerDistributor
from .trigger_sorting import get_seq_trigger_info, sort_sequencers

from qblox_instruments import (
    InstrumentClass, InstrumentType,
    SystemStatuses, SystemStatus, SystemStatusSlotFlags,
)


logger = logging.getLogger(__name__)


class Q1Simulator(qc.Instrument, Q1Module):
    _pulsar_parameters = [
        'reference_source',
    ]
    _log_only_params = [
        'led_brightness',
    ]

    def __init__(self, name, n_sequencers=6, sim_type=None):
        check_qblox_instrument_version()
        super().__init__(name)
        super().init_module(n_sequencers, sim_type)

        for par_name in self._pulsar_parameters:
            self.add_parameter(par_name, set_cmd=partial(self._set, par_name))

        for par_name in self._log_only_params:
            self.add_parameter(par_name,
                               set_cmd=partial(self._log_set, par_name))
        self.ignore_triggers = False

    def get_idn(self):
        return dict(vendor='Q1Simulator', model=self._sim_type, serial='', firmware='')

    @property
    def instrument_class(self):
        return InstrumentClass.PULSAR

    @property
    def instrument_type(self):
        return InstrumentType[self._sim_type]

    def reset(self):
        self.invalidate_cache()
        super().reset()

    def get_system_status(self):
        return SystemStatus(
            SystemStatuses.OKAY,
            [],
            SystemStatusSlotFlags({}))

    def start_sequencer(self, sequencer: int | None = None):
        if sequencer is not None:
            self.sequencers[sequencer].run()
            return

        # Get list of armed sequencers
        # pass to sequence executor

        sequencers = [self.sequencers[seq_number] for seq_number in self.armed_sequencers]
        run_sequencers(sequencers, self.ignore_triggers)

    def _log_set(self, name, value):
        logger.info(f'{self.name}: {name}={value}')

    def plot(self,
             t_min: float = None,
             t_max: float = None,
             channels: list[str] | list[int] | None = None,
             analogue_filter: bool = False,
             analogue_output_frequency: float = 4e9,
             output_per_sequencer: bool = True,
             **kwargs):
        """Plots the simulated output of the module.

        Args:
            t_min: minimum time in the plot.
            t_max: maximum time in the plot.
            channels: If not None specifies the channels to plot by name or sequencer number.
            analogue_filter: plot result after applying (estimated) analog filter.
            analogue_output_frequency: sample rate of analogue output
            output_per_sequencer:
                if True: plot data for individual sequencers.
                if False: plot data for physical front panel output.
        """
        pt.figure()
        super().plot(t_min=t_min, t_max=t_max, channels=channels,
                     analogue_filter=analogue_filter,
                     analogue_output_frequency=analogue_output_frequency,
                     output_per_sequencer=output_per_sequencer)
        pt.grid(True)
        pt.legend()
        pt.xlabel('[ns]')
        pt.ylabel('[V]')
        pt.show()


def run_sequencers(sequencers: list[Q1Sequencer], ignore_triggers=False):
    if ignore_triggers:
        for seq in sequencers:
            seq.run()
        return

    # sort on used triggers

    # TODO Refactor trigger distribution in runtime distribution.
    seq_infos = [get_seq_trigger_info(seq) for seq in sequencers]
    trigger_dist = TriggerDistributor()
    seq_infos = sort_sequencers(seq_infos)
    for seq_info in seq_infos:
        seq = seq_info.sequencer
        seq.set_trigger_events(trigger_dist.get_trigger_events())
        seq.run()
        trigger_dist.add_emitted_triggers(seq.get_acq_trigger_events())
