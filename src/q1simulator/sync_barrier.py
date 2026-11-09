from threading import Condition


class SyncBarrier:
    def __init__(self):
        self._sequencers: set[str] = set()
        self._condition = Condition()
        self._sync_set = _SyncSet(self._sequencers)
        self._abort_sequencers: set[str] = set()

    def add_sequencer(self, sequencer_name: str):
        self._sequencers.add(sequencer_name)
        self._abort_sequencers.discard(sequencer_name)

    def remove_sequencer(self, sequencer_name: str):
        with self._condition:
            self._sequencers.discard(sequencer_name)
            self._abort_sequencers.discard(sequencer_name)
            self._condition.notify_all()

    def abort_sequencer(self, sequencer_name: str):
        with self._condition:
            self._abort_sequencers.add(sequencer_name)
            self._condition.notify_all()

    def wait_sync(self, sequencer_name: str, ref_time: int) -> int:

        with self._condition:
            sync_set = self._sync_set
            sync_set.add(sequencer_name, ref_time)
            if sync_set.is_complete:
                # create new empty set for next sync.
                self._sync_set = _SyncSet(self._sequencers)
                self._condition.notify_all()
            else:
                while not sync_set.is_complete and sequencer_name not in self._abort_sequencers:
                    self._condition.wait()
                if not sync_set.is_complete:
                    # aborted: leave the sync set for the next run.
                    sync_set.remove(sequencer_name)
                    # return ref_time to avoid negative time jump.
                    return ref_time

            return sync_set.exit_time


class _SyncSet:
    def __init__(self, sequencers: set[str]):
        self._sequencers = sequencers
        self._waiting: set[str] = set()
        self._ref_times: dict[str, int] = {}

    def add(self, sequencer_name: str, ref_time: int):
        self._waiting.add(sequencer_name)
        self._ref_times[sequencer_name] = ref_time

    def remove(self, sequencer_name: str):
        self._waiting.discard(sequencer_name)
        try:
            del self._ref_times[sequencer_name]
        except KeyError:
            pass

    @property
    def is_complete(self):
        return self._waiting == self._sequencers

    @property
    def exit_time(self):
        # Add 200 ns for sync overhead
        return max(self._ref_times.values()) + 200
