import logging
from collections import defaultdict
from dataclasses import dataclass
from threading import Condition

from .sync_barrier import SyncBarrier


logger = logging.getLogger(__name__)


@dataclass
class BaseEvent:
    event_time: int
    event_type: int
    """event type is used for sorting. @@@ might not be needed after all."""


@dataclass
class TriggerEvent(BaseEvent):
    address: int
    state: int
    """0 or 1. The HW only sends a 1. 0 is not transmitted. We do it for debugging."""


@dataclass
class FeedbackEvent(BaseEvent):
    event_id: int
    value: int  # 32 bit result


class SequencerQueue:
    def __init__(self):
        self._queue: list[BaseEvent] = []

    def add_trigger(self, time: int, address: int, state: int):
        self._append(TriggerEvent(time, 0, address, state))

    def add_feedback_event(self, time: int, event_id, data: list[int]):
        for value in data:
            self._append(FeedbackEvent(time, 1, event_id, value))

    def _append(self, event: BaseEvent):
        self._queue.append(event)
        self._queue.sort(key=lambda e: (e.event_time, e.event_type))

    def get_event(self, max_time: int) -> BaseEvent | None:
        if len(self._queue) == 0:
            return None
        event = self._queue[0]
        if event.event_time <= max_time:
            self._queue.pop(0)
            return event
        else:
            return None


class EventDistributor:
    def __init__(self):
        self._condition = Condition()
        self._event_targets: dict[int, set[str]] = defaultdict(default_factory=set)
        self._sync_barrier = SyncBarrier()
        self._sequencer_times: dict[str, SequencerTime] = {}
        # event and trigger queue per sequencer.
        self._sequencer_queue: dict[str, SequencerQueue] = {}
        self._abort_sequencers: set[str] = set()
        self._min_time: int = 0
        self._next_emission: int = 0
        self._abort = False

    def add_event_receiver(self, sequencer_name: str, event_id: int):
        self._event_targets[event_id].add(sequencer_name)

    def set_sequencer_sync_en(self, sequencer_name: str, synced: bool):
        if synced:
            self._sync_barrier.add_sequencer(sequencer_name)
        else:
            self._sync_barrier.remove_sequencer(sequencer_name)

    def arm_sequencer(self, sequencer_name: str, synced: bool):
        self._sequencer_times[sequencer_name] = SequencerTime()
        self._sequencer_queue[sequencer_name] = SequencerQueue()
        self._abort_sequencers.discard(sequencer_name)

    def start_sequencer(self, sequencer_name: str):
        self._sequencer_times[sequencer_name].update(rt_time=0, system_time=self._get_ref_time())

    def stop_sequencer(self, sequencer_name: str):
        with self._condition:
            try:
                del self._sequencer_times[sequencer_name]
            except KeyError:
                pass
            self._update_min_time()
            self._condition.notify_all()

    def abort_sequencer(self, sequencer_name: str):
        self._sync_barrier.abort_sequencer(sequencer_name)
        with self._condition:
            self._abort_sequencers.add(sequencer_name)
            self._condition.notify_all()

    def wait_sync_sequencer(self, sequencer_name: str, rt_time: int) -> int:
        """
        Waits for all sequencers to synchronize.
        Returns new RT time
        """
        self._update_sequencer_time(sequencer_name, rt_time=rt_time)
        t_start = self._sequencer_times[sequencer_name].system_time

        # Wait for sync and get new absolute sequencer time.
        t_synced = self._sync_barrier.wait_sync(sequencer_name, t_start)
        new_rt_time = rt_time + t_synced - t_start

        self._update_sequencer_time(sequencer_name, rt_time=new_rt_time)
        return new_rt_time

    def set_sequencer_time(self, sequencer_name: str, rt_time: int):
        """
        The sequencer time should be set regularly by running sequencers to prevent starvation of other sequencers.
        Update interval could be something like once every 10 instructions.
        """
        if self._abort:
            logger.info(f"Abort {sequencer_name}")
            raise KeyboardInterrupt(f"Aborting {sequencer_name}")
        self._update_sequencer_time(sequencer_name, rt_time=rt_time)

    def abort(self):
        logger.info("Request abort ALL")
        with self._condition:
            self._abort = True
            self._condition.notify_all()
        self._sync_barrier.abort()

    def _update_sequencer_time(self, sequencer_name: str, /,
                               rt_time: int | None = None,
                               sys_time: int | None = None):
        with self._condition:
            if rt_time is not None:
                self._sequencer_times[sequencer_name].update(rt_time=rt_time)
            if sys_time is not None:
                self._sequencer_times[sequencer_name].update(system_time=sys_time)
            self._update_min_time()
            self._condition.notify_all()

    def _update_min_time(self):
        if not self._sequencer_times:
            self._min_time = 0
        else:
            self._min_time = min(seq.system_time for seq in self._sequencer_times.values())

    def _wait_till(self, sequencer_name: str, system_time: int):
        with self._condition:
            wait = self._min_time < system_time
            if wait:
                logger.info(f"Sequencer {sequencer_name} waits...")
            while self._min_time < system_time and not self._abort and sequencer_name not in self._abort_sequencers:
                self._condition.wait()
            if wait:
                logger.info(f"Sequencer {sequencer_name} continues")

    def get_event(self, sequencer_name: str, rt_time: int) -> FeedbackEvent | TriggerEvent | None:
        """
        Returns next event from event queue with delivery time <= rt_time.

        Note: for feedback events the event time is used to update the Q1Core clock.

        Returns:
            event_time, event
        """
        self.set_sequencer_time(sequencer_name, rt_time)
        sys_time = self._sequencer_times[sequencer_name].system_time
        # wait till all sequencers are at or beyond this system time.
        self._wait_till(sequencer_name, sys_time)

        return self._sequencer_queue[sequencer_name].get_event(sys_time)


        # TODO every render/wait step in RT
        # return with time of event reception
        # similar for triggers...

        # fb_pull sets flag to retrieve events.
        # active condition sets flag to retrieve events.

        # if fb_pull flag set: call get_event and update Q1Core time with event time.
        # If none: Underflow


    def emit_trigger(self, sequencer_name: str, rt_time: int, address: int, state: int):
        self.set_sequencer_time(sequencer_name, rt_time)
        sys_time = self._sequencer_times[sequencer_name].system_time
        t_delivery = sys_time + 250
        for sequencer_queue in self._sequencer_queue.values():
            sequencer_queue.add_trigger(t_delivery, address, state)

        # TDOO sequencer: process new triggers update counts and trigger thresholds to mask at every conditional instruction


    def fb_send(self, sequencer_name: str, rt_time: int, event_id: int, data: list[int], data_type: str):
        """
        Note: Distribution latencies are not exact.

        Deliveries are multi-cast or self-cast. Intra-cast is currently handled as multi-cast.
        """

        # distribution latency for self-cast for 1 32 bit value.
        data_type_latency = {
            "tb": 160,
            "iq": 160,  # 4 or 20 ns is added for the 2nd byte
            "q1": 60,
            }

        if event_id == 0:
            logger.info("Dropping event with event_id == 0")
            return

        self.set_sequencer_time(sequencer_name, rt_time)
        sys_time = self._sequencer_times[sequencer_name].system_time

        type_latency = data_type_latency[data_type]

        length = len(data)
        # TODO special case: self cast.
        if event_id <= 16:
            latency = 4*(length-1) + type_latency
            t_delivery = sys_time + latency
            sequencer_queue = self._sequencer_queue[sequencer_name]
            sequencer_queue.add_feedback_event(t_delivery, event_id, data)
        else:
            if self._next_emission > sys_time:
                t_send = self._next_emission
            else:
                t_send = sys_time
            # just add 200 ns busy occupation time.
            self._next_emission = t_send + 200
            # multi-cast latency is ~320 + type latency.
            latency = 20*(length-1) + type_latency + 320
            t_delivery = t_send + latency

            targets = self._event_targets[event_id]
            for target_name in targets:
                sequencer_queue = self._sequencer_queue[target_name]
                sequencer_queue.add_feedback_event(t_delivery, event_id, data)

    def _get_ref_time(self):
        return max(seq.system_time for seq in self._sequencer_times.values())


class SequencerTime:
    def __init__(self):
        self._sync_offset: int = 0
        self._rt_time: int = 0

    def update(self, /, rt_time: int | None = None, system_time: int | None = None):
        if rt_time is not None:
            self._rt_time = rt_time
        if system_time is not None:
            self._sync_offset = system_time - self._rt_time

    @property
    def system_time(self):
        return self._sync_offset + self._rt_time

    @property
    def rt_time(self):
        return self._rt_time
