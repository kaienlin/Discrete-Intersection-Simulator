import enum

class EdgeType(enum.Enum):
    TYPE_1 = enum.auto()
    TYPE_2 = enum.auto()
    TYPE_3 = enum.auto()


class Edge:
    default_waiting_time = {
        EdgeType.TYPE_1: 0,
        EdgeType.TYPE_2: 2,
        EdgeType.TYPE_3: 3
    }
    def __init__(
        self,
        _id: int,
        v_from: object,
        v_to: object,
        type: EdgeType,
        waiting_time: int = -1
    ) -> None:
        self.id: int = _id
        self.v_from: object = v_from
        self.v_to: object = v_to
        self.type: EdgeType = type
        if waiting_time == -1:
            self.waiting_time = self.default_waiting_time[type]
        else:
            self.waiting_time = waiting_time

