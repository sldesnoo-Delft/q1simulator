import logging
from typing import List
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SequencerTriggerInfo:
    module: int
    seq_number: int
    trigger_addr: int
    ''' address value of emitted trigger:  '''
    used_triggers: int
    ''' bits for all triggers used by sequencer conditions.
    0b00001010 for triggers 2 and 4.
    '''

def get_seq_trigger_info(module, seq_number, sequencer):
    if hasattr(sequencer,'thresholded_acq_trigger_en') and sequencer.thresholded_acq_trigger_en():
        trigger_addr = sequencer.thresholded_acq_trigger_address()
    else:
        trigger_addr = 0
    return SequencerTriggerInfo(
            module, seq_number,
            trigger_addr,
            sequencer.get_used_triggers()
            )

def sort_sequencers(sequencers: List[SequencerTriggerInfo]):
    seq_in = sequencers.copy()
    seq_out = []

    # check if there are not 2 sequencers with same emitted trigger.
    emitters = defaultdict(int)
    for seq in seq_in:
        if seq.trigger_addr:
            emitters[seq.trigger_addr] += 1
            if emitters[seq.trigger_addr] > 1:
                logger.warning(f'Multiple sequencers emit trigger {seq.trigger_addr}')

    emitted = 0
    while len(seq_in):
        added = 0
        for seq in seq_in.copy():
            seq_requires = seq.used_triggers
            seq_emits = 1 << (seq.trigger_addr - 1) if seq.trigger_addr else 0
            if seq_emits and emitters[seq.trigger_addr] == 1:
                # It's the only emitter, so it can use it's own output.
                seq_requires &= ~seq_emits
            if (seq_requires & ~emitted) == 0:
                emitted |= seq_emits
                seq_out.append(seq)
                seq_in.remove(seq)
                added += 1
        if added == 0:
            raise Exception(f"sorting failed... {seq_out}")

    return seq_out
