from enum import Enum

class ProbeType(Enum):
    POSITIVE = "positive"
    ADVERSARIAL = "adversarial"
    EDGE_CASE = "edge_case"