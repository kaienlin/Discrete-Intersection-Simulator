from abc import ABC, abstractmethod


class Policy(ABC):
    @abstractmethod
    def decide(self, state: int) -> int:
        return 0
