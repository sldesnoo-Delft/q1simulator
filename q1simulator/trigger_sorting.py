import logging
from typing import List
from dataclasses import dataclass

from .q1sequencer import Q1Sequencer


logger = logging.getLogger(__name__)

@dataclass
class SequencerTriggerInfo:
    sequencer: Q1Sequencer
    emitted_triggers: int
    ''' bits for emitted trigger.
    0b00001010 for triggers 2 and 4.
    '''
    used_triggers: int
    ''' bits for all triggers used by sequencer conditions.
    0b00001010 for triggers 2 and 4.
    '''

def get_seq_trigger_info(sequencer):
    if hasattr(sequencer,'thresholded_acq_trigger_en') and sequencer.thresholded_acq_trigger_en():
        emitted_triggers = 1 << (sequencer.thresholded_acq_trigger_address() - 1)
    else:
        emitted_triggers = 0
    return SequencerTriggerInfo(
            sequencer,
            emitted_triggers,
            sequencer.get_used_triggers()
            )

def get_emitted_triggers(sequencers: List[SequencerTriggerInfo]):
    emitted = 0
    for seq in sequencers:
        emitted |= seq.emitted_triggers
    return emitted

def sort_sequencers(sequencers: List[SequencerTriggerInfo]):
    ''' Sorts sequencers putting trigger producers before trigger consumers.

    Sequencers are executed sequentially. A sequencer must be executed after
    all sequencers that emit triggers that it uses.

    Notes:
        * A sequencer may use a trigger not emitted by any sequencer.
        * A sequencer may use a trigger that it also emits.
    '''
    result = []
    unsorted = sequencers.copy()
    while len(unsorted) > 0:
        n_start = len(unsorted)
        to_be_emitted = get_emitted_triggers(unsorted)
        for seq in unsorted.copy():
            if seq.used_triggers & to_be_emitted == 0:
                # sequence does not use triggers that will be emitted after it.
                result.append(seq)
                unsorted.remove(seq)
            elif seq.used_triggers & to_be_emitted & ~seq.emitted_triggers == 0:
                # sequence emits a triggers it uses. Check if there are other sequencers emitting the trigger.
                remaining = unsorted.copy()
                remaining.remove(seq)
                rem_emitted = get_emitted_triggers(remaining)
                if seq.used_triggers & rem_emitted == 0:
                    # OK. It does not use triggers of remaining sequencers.
                    result.append(seq)
                    unsorted.remove(seq)

        if len(unsorted) == n_start:
            raise Exception(f"Sorting sequencers on triggers failed. Unsorted: {unsorted}")

    return result
