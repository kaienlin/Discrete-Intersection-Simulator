from base64 import decode
from typing import Optional
import enum


class EdgeType(enum.Enum):
    TYPE_1 = enum.auto()
    TYPE_2 = enum.auto()
    TYPE_3 = enum.auto()
    TYPE_4 = enum.auto()


class Edge:
    default_waiting_time = {
        EdgeType.TYPE_1: 1,
        EdgeType.TYPE_2: 2,
        EdgeType.TYPE_3: 3
    }
    def __init__(
        self,
        _id: int,
        v_from: object,
        v_to: object,
        edge_type: EdgeType,
        waiting_time: Optional[int] = None,
        decided: bool = True
    ) -> None:
        self.id: int = _id
        self.v_from: object = v_from
        self.v_to: object = v_to
        self.type: EdgeType = edge_type
        self.decided: bool = decided

        if waiting_time is None:
            if edge_type == EdgeType.TYPE_4:
                raise Exception("Type-4 edge construction should be given a waiting time")
            self.waiting_time = self.default_waiting_time[edge_type]
        else:
            self.waiting_time = waiting_time
