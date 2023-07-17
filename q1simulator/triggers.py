from copy import copy
import math
from dataclasses import dataclass
from typing import List

@dataclass
class TriggerEvent:
    '''
    Trigger even data.
    Note:
        The simulator also sends events with state == 0.
        This allows worst case timing checks.
    '''
    addr: int
    time: int # time in ns
    state: bool

    def with_delay(self, delay=220, alignment=28):
        res = self.copy()
        res.time = math.ceil(res.time / alignment) * alignment
        res.time += delay
        return res

    def copy(self):
        return copy(self)

# TODO Refactor trigger distributor into runtime distributor
# Run till sequencer calls set_cond. Block it and let others run.
# Sequencer may resume when all other sequencers are at least at equal time.

class TriggerDistributor:
    def __init__(self):
        self.trigger_events: List[TriggerEvent] = []

    def add_emitted_triggers(self, trigger_events):
        # Assumes there are no collisions on the trgigger network.

        delayed_events = [e.with_delay() for e in trigger_events]

        self.trigger_events += delayed_events
        self.trigger_events.sort(key=lambda e:e.time)

    def get_trigger_events(self):
        return self.trigger_events.copy()

