# metric_result.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class MetricResult:
    metric: str
    qid: Optional[str]
    score: float
    details: Dict[str, Any]  # fully structured; safe to json.dumps