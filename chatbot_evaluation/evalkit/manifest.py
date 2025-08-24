from typing import Dict, Any
import os
from .files import sha256_file, sha256_text, count_jsonl, utc_ts, git_commit

def _prompt_entries(prompts_cfg: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    out = {}
    for metric, mapping in prompts_cfg.items():
        out[metric] = {}
        for key, path in mapping.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Prompt file not found: {path}")
            out[metric][key] = {
                "file": path,
                "sha256": sha256_file(path),
            }
    return out

def build_manifest(seed: Dict[str, Any], code_hash: str | None = None) -> Dict[str, Any]:
    ds = seed["dataset"]
    dspath = ds["path"]
    ds_sha = sha256_file(dspath)
    ds_size = count_jsonl(dspath)

    run_id = f'{utc_ts().replace(":","-").replace("Z","Z")}_{seed["model_under_test"]["name"]}'
    commit = git_commit()

    manifest = {
        "schema_version": seed.get("schema_version","1.0.0"),
        "run": {
            "id": run_id,
            "started_at": utc_ts(),
            "git_commit": commit,
            "code_sha256": code_hash or "",
        },
        "dataset": {
            "name": ds["name"],
            "path": dspath,
            "sha256": ds_sha,
            "split": ds.get("split","eval"),
            "size": ds_size,
        },
        "model_under_test": seed["model_under_test"],
        "retrieval": seed.get("retrieval", {}),
        "judges": seed.get("judges", {}),
        "prompts": _prompt_entries(seed.get("prompts", {})),
        "metrics": seed.get("metrics", []),
        "reporting": seed.get("reporting", {}),
        "notes": seed.get("notes",""),
    }
    return manifest