from threading import Condition


class SyncBarrier:
    def __init__(self):
        self._sequencers: set[str] = set()
        self._condition = Condition()
        self._sync_set = _SyncSet(self._sequencers)
        self._abort_sequencers: set[str] = set()
        self._abort = False

    def add_sequencer(self, sequencer_name: str):
        self._sequencers.add(sequencer_name)
        self._abort_sequencers.discard(sequencer_name)

    def remove_sequencer(self, sequencer_name: str):
        self._sequencers.discard(sequencer_name)
        self._abort_sequencers.discard(sequencer_name)

    def abort_sequencer(self, sequencer_name: str):
        self._abort_sequencers.add(sequencer_name)

    def wait_sync(self, sequencer_name: str, ref_time: int) -> int:

        with self._condition:
            sync_set = self._sync_set
            sync_set.add(sequencer_name, ref_time)
            if sync_set.is_complete:
                # create new empty set for next sync.
                self._sync_set = _SyncSet(self._sequencers)
                self._condition.notify_all()
            else:
                while not sync_set.is_complete and not self._abort and sequencer_name not in self._abort_sequencers:
                    self._condition.wait()
            return sync_set.exit_time

    def abort(self):
        with self._condition:
            self._abort = True
            self._condition.notify_all()


class _SyncSet:
    def __init__(self, sequencers: set[str]):
        self._sequencers = sequencers
        self._waiting: set[str] = set()
        self._ref_times: list[int] = []

    def add(self, sequencer_name: str, ref_time: int):
        self._waiting.add(sequencer_name)
        self._ref_times.append(ref_time)

    @property
    def is_complete(self):
        return self._waiting == self._sequencers

    @property
    def exit_time(self):
        # Add 200 ns for sync overhead
        return max(self._ref_times) + 200
