from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, List
from .files import read_yaml, sha256_file, count_jsonl, write_yaml, utc_ts, git_commit

def _resolve_id_to_path(root: Path, typed_id: str, leaf: str) -> Path:
    # "faithfulness/nli_judge/fa_IR@1.4.1" -> prompts/faithfulness/nli_judge/fa_IR/v1.4.1/<leaf>
    typ, role, locale_ver = typed_id.split("/", 2)  # metric family path or dataset group
    if "@" not in locale_ver:
        raise ValueError(f"Missing @version in id: {typed_id}")
    locale, ver = locale_ver.split("@")
    ver = "v" + ver
    if root.name == "prompts":
        p = root / typ / role / locale / ver / leaf
    elif root.name == "datasets":
        domain, name_ver = typ, role + "/" + locale_ver  # reuse fields
        name, ver = name_ver.split("@")
        ver = "v" + ver
        p = root / domain / name / ver / leaf
    elif root.name == "metrics":
        name, ver = (typ, role.split("@")[1]) if "@" in role else (typ, "v0.0.0")
        ver = "v" + ver if not ver.startswith("v") else ver
        p = root / name / role.split("@")[0] / ver
    elif root.name == "retrieval":
        name, ver = (typ, role.split("@")[1]) if "@" in role else (typ, "v0.0.0")
        ver = "v" + ver if not ver.startswith("v") else ver
        p = root / name / ver / leaf
    elif root.name == "judges":
        kind, name_ver = typ, role
        name, ver = name_ver.split("@")
        ver = "v" + ver
        p = root / kind / name / ver / leaf
    else:
        raise ValueError("Unknown root")
    return p

def resolve_suite(suite_path: str) -> dict:
    root = Path(suite_path).parent.parent  # repo root if suite at config/suite.yaml
    suite = read_yaml(suite_path)

    # datasets
    ds_entries = []
    for ds_id in suite.get("datasets", []):
        path = _resolve_id_to_path(root / "datasets", ds_id["id"], "data.jsonl")
        ds_entries.append({
            "id": ds_id["id"],
            "path": str(path),
            "sha256": sha256_file(str(path)),
            "size": count_jsonl(str(path)),
        })

    # retrieval bundle
    rb_id = suite["retrieval_bundle"]
    rb_path = _resolve_id_to_path(root / "retrieval", rb_id, "bundle.yaml")
    rb_entry = {"id": rb_id, "file": str(rb_path), "sha256": sha256_file(str(rb_path))}

    # judges
    llm_id = suite["judges"]["llm"]
    llm_path = _resolve_id_to_path(root / "judges", "llm/"+llm_id, "config.yaml")
    emb_id = suite["judges"]["embeddings"]
    emb_path = _resolve_id_to_path(root / "judges", "embeddings/"+emb_id, "config.yaml")

    # prompts
    prompts = {}
    for metric, roles in suite.get("prompts", {}).items():
        prompts[metric] = {}
        for role, pid in roles.items():
            p = _resolve_id_to_path(root / "prompts", pid, "instruction.md")
            prompts[metric][role] = {"id": pid, "file": str(p), "sha256": sha256_file(str(p))}

    # metrics
    metrics = []
    for m in suite.get("metrics", []):
        impl_id = m["impl"]
        # record impl id; code provenance is captured by git commit in MANIFEST + lock below
        metrics.append({
            "name": m["name"],
            "impl_id": impl_id,
            "params": m.get("params", {}),
            "datasets": m.get("datasets", [d["id"] for d in suite["datasets"]]),
        })

    manifest = {
        "schema_version": "1.0.0",
        "run": {
            "id": f'{utc_ts().replace(":","-")}_{suite["name"]}',
            "started_at": utc_ts(),
            "git_commit": git_commit(),
        },
        "suite": {"name": suite["name"], "version": suite.get("suite_version",1), "notes": suite.get("notes","")},
        "datasets": ds_entries,
        "model_under_test": suite["model_under_test"],
        "retrieval": {"bundle": rb_entry},
        "judges": {
            "llm": {"id": llm_id, "file": str(llm_path), "sha256": sha256_file(str(llm_path))},
            "embeddings": {"id": emb_id, "file": str(emb_path), "sha256": sha256_file(str(emb_path))},
        },
        "prompts": prompts,
        "metrics": metrics,
        "reporting": suite.get("reporting", {}),
        "output": suite.get("output", {"base_dir":"runs"}),
    }

    # lockfile adds environment pin
    lock = {
        "manifest_run_id": manifest["run"]["id"],
        "python": os.popen("python -V").read().strip(),
        "pip_freeze": os.popen("pip freeze").read().splitlines(),
        "ragas_version": os.popen("python -c 'import ragas,sys;print(ragas.__version__)' 2>/dev/null || echo 'NA'").read().strip(),
        "deepeval_version": os.popen("python -c 'import deepeval,sys;print(deepeval.__version__)' 2>/dev/null || echo 'NA'").read().strip(),
    }
    return {"manifest": manifest, "lock": lock}

def write_manifest_and_lock(out_dir: str, suite_path: str):
    from .files import ensure_dir, write_yaml
    ensure_dir(out_dir)
    r = resolve_suite(suite_path)
    write_yaml(os.path.join(out_dir, "manifest.yaml"), r["manifest"])
    write_yaml(os.path.join(out_dir, "lock.yaml"), r["lock"])
    return r["manifest"], r["lock"]