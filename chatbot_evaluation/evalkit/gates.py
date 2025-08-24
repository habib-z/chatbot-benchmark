import math
import operator, re
from typing import Dict, Any

OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}

def evaluate_gates(agg: Dict[str, Any], gate_rules: Dict[str, str]) -> Dict[str, bool]:
    """
    Gate rules are like:
      {
        "faithfulness_precision_micro": ">= 0.95",
        "factual_f1_micro": ">= 0.85"
      }
    We also accept aliases without suffix; if missing, we try _micro then _macro.
    """
    out: Dict[str, bool] = {}
    metrics = agg.get("metrics", {})
    for key, expr in gate_rules.items():
        m = re.match(r"\s*(>=|<=|>|<|==)\s*([0-9.]+)\s*$", expr)
        if not m:
            out[key] = False
            continue
        op, num = m.group(1), float(m.group(2))

        # resolve key
        if key in metrics:
            k = key
        elif f"{key}_micro" in metrics:
            k = f"{key}_micro"
        elif f"{key}_macro" in metrics:
            k = f"{key}_macro"
        else:
            # old fallback: some folks used "faithfulness_mean", etc.
            k = key.replace("mean", "_mean")

        val = metrics.get(k, None)
        out[key] = OPS[op](float(val), num) if (val is not None and not math.isnan(val)) else False
    return out