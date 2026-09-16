from abc import ABC, abstractmethod
from threading import Thread

from .event_distributor import EventDistributor


class Task(ABC):
    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def run(self):
        ...


class Scheduler:
    def __init__(self, event_distributor: EventDistributor):
        self.event_distributor = event_distributor
        self._running_sequencers: list[str] = []
        self._sequencer_threads: dict[str, Thread] = {}

    def start_sequencer(self, sequencer: Task):
        name = sequencer.name
        # try to remove old threads. Just to be sure.
        self.join_sequencer(name)
        if name in self._running_sequencers or name in self._sequencer_threads:
            raise Exception(f"Sequencer {name} already running")

        self._running_sequencers.append(name)
        thread = Thread(target=sequencer.run, name=name)
        thread.start()
        self._sequencer_threads[name] = thread

    def join_sequencer(self, sequencer_name: str) -> bool:
        # sequencer must be in stopped state to join.
        if sequencer_name not in self._sequencer_threads:
            return True
        thread = self._sequencer_threads[sequencer_name]
        thread.join(0)
        alive = thread.is_alive()
        if not alive:
            del self._sequencer_threads[sequencer_name]
            self._running_sequencers.remove(sequencer_name)
        return not alive
