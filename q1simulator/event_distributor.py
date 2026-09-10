import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from threading import Condition

from .sync_barrier import SyncBarrier


logger = logging.getLogger(__name__)


class EventType(Enum):
    TRIGGER = 0
    FEEDBACK = 1
    FB_WRITECOMBINE = 2


@dataclass
class BaseEvent:
    event_time: int
    event_type: EventType
    """event type is used for sorting. """


@dataclass
class TriggerEvent(BaseEvent):
    address: int
    state: int
    """0 or 1. The HW only sends a 1. 0 is not transmitted. We do it for debugging."""


@dataclass
class FeedbackEvent(BaseEvent):
    event_id: int
    values: list[int]  # 32 bit results
    event_core_time: int | None = None


class SequencerQueue:
    def __init__(self):
        self._queue: list[BaseEvent] = []

    def clear(self):
        self._queue = []

    def add_trigger(self, time: int, address: int, state: int):
        self._append(TriggerEvent(time, EventType.TRIGGER, address, state))

    def add_feedback_event(self, time: int, event_id, data: list[int], write_combine: bool):
        event_type = EventType.FB_WRITECOMBINE if write_combine else EventType.FEEDBACK
        self._append(FeedbackEvent(time, event_type, event_id, data))

    def _append(self, event: BaseEvent):
        self._queue.append(event)
        self._queue.sort(key=lambda e: (e.event_time, e.event_type))

    def get_event(self, max_time: int) -> BaseEvent | None:
        if len(self._queue) == 0:
            return None
        q = self._queue
        event = q[0]

        if event.event_time <= max_time:
            q.pop(0)
            if event.event_type == EventType.FB_WRITECOMBINE:
                logger.info(f"Combining WC: {event}")
                while (q and q[0].event_time == event.event_time and q[0].event_type == EventType.FB_WRITECOMBINE
                       and q[0].event_id == event.event_id):
                    combine = q.pop(0)
                    if len(event.values) != len(combine.values):
                        raise Exception(f"Unequal length for feedback event with id {event.event_id}")
                    # bitwise or of data
                    event.values = [d1 | d2 for d1, d2 in zip(event.values, combine.values)]
                logger.info(f"Combined {event.event_id} ({event.event_time}): {event.values}")
            else:
                logger.info(f"Event {event.event_id} ({event.event_time}): {event.values}")
            return event
        else:
            logger.debug(f"No event {max_time}")
            return None


class EventDistributor:
    def __init__(self):
        self._condition = Condition()
        self._event_targets: dict[int, set[str]] = defaultdict(set)
        self._sync_barrier = SyncBarrier()
        self._sequencer_times: dict[str, SequencerTime] = {}
        # event and trigger queue per sequencer.
        self._sequencer_queue: dict[str, SequencerQueue] = {}
        self._abort_sequencers: set[str] = set()
        self._min_time: int = 0
        self._abort = False

    def clear_router(self, sequencer_name: str | None = None):
        if sequencer_name is None:
            self._event_targets.clear()
            for queue in self._sequencer_queue.values():
                queue.clear()
        else:
            for target in self._event_targets.values():
                target.discard(sequencer_name)
            self._sequencer_queue[sequencer_name].clear()

    def set_route(self, event_id: int, sequencer_name: str):
        self._event_targets[event_id].add(sequencer_name)
        logger.info(f"set_route {event_id} -> {sequencer_name}; ({self._event_targets[event_id]})")

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
        self._sequencer_times[sequencer_name].start(self._get_ref_time())

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
                logger.info(f"Sequencer {sequencer_name} waits at sys:{system_time} ...")
            while self._min_time < system_time and not self._abort and sequencer_name not in self._abort_sequencers:
                self._condition.wait()
            if wait:
                logger.info(f"Sequencer {sequencer_name} continues")
            if self._abort or sequencer_name not in self._abort_sequencers:
                logger.info(f"Sequencer abort ({self._abort}, {sequencer_name in self._abort_sequencers})")

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

        event = self._sequencer_queue[sequencer_name].get_event(sys_time)
        if isinstance(event, FeedbackEvent):
            event.event_core_time = event.event_time - self._sequencer_times[sequencer_name].offset

        return event

    def emit_trigger(self, sequencer_name: str, rt_time: int, address: int, state: int):
        self.set_sequencer_time(sequencer_name, rt_time)
        sys_time = self._sequencer_times[sequencer_name].system_time
        t_delivery = sys_time + 250
        for sequencer_queue in self._sequencer_queue.values():
            sequencer_queue.add_trigger(t_delivery, address, state)

    def fb_send(self, sequencer_name: str, rt_time: int, event_id: int, data: list[int], data_type: str,
                write_combine: bool):
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
        logger.info(f"fb send: {sequencer_name}, {rt_time} ({sys_time}), {event_id}, {data}")
        self._wait_till(sequencer_name, sys_time)

        type_latency = data_type_latency[data_type]

        length = len(data)
        if event_id <= 15:
            # special case: self cast.
            latency = 4*(length-1) + type_latency
            t_delivery = sys_time + latency
            sequencer_queue = self._sequencer_queue[sequencer_name]
            sequencer_queue.add_feedback_event(t_delivery, event_id, data)
        else:
            # TODO: distinct intra-cast / multi-cast
            # Bus occupancy is currently ignored!
            t_send = sys_time

            # Note: Write combines are merged upon reception

            # multi-cast latency is ~320 + type latency.
            latency = 20*(length-1) + type_latency + 320
            t_delivery = t_send + latency

            targets = self._event_targets[event_id]
            for target_name in targets:
                logger.debug(f"event {event_id} -> {target_name} at {t_delivery}(sys) latency:{latency}")
                sequencer_queue = self._sequencer_queue[target_name]
                sequencer_queue.add_feedback_event(t_delivery, event_id, data, write_combine)

    def _get_ref_time(self):
        return max(seq.system_time for seq in self._sequencer_times.values())

    def _print(self, sequencer_name):
        print("Sequencer", sequencer_name)
        print(self._sequencer_times[sequencer_name])
        print(self._sequencer_queue[sequencer_name]._queue)
        print("Abort", self._abort, sequencer_name in self._abort_sequencers)


class SequencerTime:
    def __init__(self):
        self._offset: int = 0
        self._rt_time: int = 0

    def start(self, offset):
        self._offset = offset
        self._rt_time: int = 0

    def update(self, /, rt_time: int | None = None):
        if rt_time is not None:
            self._rt_time = rt_time

    @property
    def system_time(self):
        return self.offset + self._rt_time

    @property
    def rt_time(self):
        return self._rt_time

    @property
    def offset(self):
        return self._offset

    def __str__(self):
        return f"sys: {self.system_time}, rt: {self.rt_time}"
