import hashlib, json, os, time, subprocess
import math
from pathlib import Path
from typing import Iterable, Any

import numpy as np


def sha256_file(path: str, buf_sz: int = 1<<20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_sz)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def count_jsonl(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f: n += 1
    return n

def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path: str, obj: dict):
    print(f"writing {path}")
    import yaml
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

# def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
#     """
#     Write JSONL (one object per line). Use this for judgments.jsonl, lockfiles with many rows, etc.
#     """
#     p = Path(path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     tmp = p.with_suffix(p.suffix + ".tmp")
#     with tmp.open("w", encoding="utf-8") as f:
#         for r in rows:
#             clean = _to_jsonable(r)
#             f.write(json.dumps(clean, ensure_ascii=False) + "\n")
#     tmp.replace(p)
def _to_jsonable(x: Any) -> Any:
    # floats (incl. numpy) -> handle NaN/Inf
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return None  # JSON doesn't support NaN/Inf
        return xf
    # ints (incl. numpy)
    if isinstance(x, (int, np.integer)):
        return int(x)
    # numpy bool
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    # numpy arrays -> lists
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    # dict
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # list/tuple
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # Path
    if isinstance(x, Path):
        return str(x)
    return x

# def write_json(path: str | Path, obj: Any) -> None:
#     """
#     Write a SINGLE JSON object (dict/list/etc.) with pretty formatting.
#     Use this for aggregates.json, gates.json, manifest.json, etc.
#     """
#     p = Path(path)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     clean = _to_jsonable(obj)
#     tmp = p.with_suffix(p.suffix + ".tmp")
#     with tmp.open("w", encoding="utf-8") as f:
#         json.dump(clean, f, ensure_ascii=False, indent=2)
#         f.write("\n")
#     tmp.replace(p)

def write_json(path, obj):
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_jsonl(path, rows):
    """
    rows: Iterable[dict]
    Writes one compact JSON object per line.
    Does NOT filter keys (e.g., keeps '@' in keys).
    Coerces numpy scalars to native types.
    """
    import json
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _coerce_scalar(v):
        try:
            import numpy as np
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.integer,)):
                return int(v)
        except Exception:
            pass
        return v

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            # ensure dict, stringify keys (keep '@'), coerce numpy scalars
            r = {str(k): _coerce_scalar(v) for k, v in dict(r).items()}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")