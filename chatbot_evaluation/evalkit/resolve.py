from __future__ import annotations
import typing as t
import hashlib
import os
from pathlib import Path
from typing import Dict, Any, List
from .files import read_yaml, sha256_file, count_jsonl, write_yaml, utc_ts, git_commit

# def _v(ver: str) -> str: return "v"+ver if not ver.startswith("v") else ver

def _v(ver: str) -> str:
    """Normalize version folder name (e.g., '1.4.1' -> 'v1.4.1')."""
    return ver if ver.startswith("v") else f"v{ver}"

def _parse_pid(pid_str: str) -> tuple[str, str, str]:
    """
    PID form: 'faithfulness/nli_judge@1.4.1'
    Returns: (family='faithfulness', role='nli_judge', ver='1.4.1')
    """
    try:
        family, rest = pid_str.split("/", 1)
        role, ver = rest.split("@", 1)
        return family, role, ver
    except Exception as e:
        raise ValueError(f"Bad pid '{pid_str}'. Expected 'family/role@ver'") from e


def sha256_file(p: str | Path) -> str:
    p = Path(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _resolve_role_dir(root: Path, role_entry: Dict[str, Any]) -> dict:
    """
    role_entry looks like: {'dir': 'benchmarks/faithfulness/statement_generator/v1'}
    We return dict with absolute paths + sha256 for instruction/examples.
    """
    if not isinstance(role_entry, dict) or "dir" not in role_entry:
        raise ValueError(f"Each role must be a dict with a 'dir' key. Got: {role_entry}")
    d = (root / role_entry["dir"]).resolve()
    instr = d / "instruction.md"
    ex = d / "examples.jsonl"
    if not instr.exists():
        raise FileNotFoundError(f"Missing instruction.md at {instr}")
    if not ex.exists():
        raise FileNotFoundError(f"Missing examples.jsonl at {ex}")
    return {
        "dir": str(d),
        "instruction": str(instr),
        "instruction_sha256": sha256_file(instr),
        "examples": str(ex),
        "examples_sha256": sha256_file(ex),
    }

def build_prompts_manifest(root: Path, suite: dict) -> dict:
    """
    From suite['prompts'] with dir entries, produce a normalized manifest.prompts.
    """
    out: dict[str, dict] = {}
    for metric, roles in (suite.get("prompts") or {}).items():
        out[metric] = {}
        # parent base_dir (optional; for display only)
        out[metric]["base_dir"] = str((root / "benchmarks" / metric).resolve())
        for role_name, role_entry in roles.items():
            out[metric][role_name] = _resolve_role_dir(root, role_entry)
    return out

def _resolve_prompts(root: Path, role_spec: t.Union[str, dict]) -> tuple[str, Path]:
    """
    role_spec can be:
      - str: 'faithfulness/nli_judge@1.4.1'
      - dict with any of:
          {'id': 'faithfulness/nli_judge@1.4.1'}
          {'id': '...', 'file': '/abs/or/rel/path/to/instruction.md'}
          {'id': '...', 'dir':  '/abs/or/rel/path/to/.../v1.4.1'}
    Returns: (pid_str, instruction_md_path)
    """
    # 1) normalize to pid_str and optional direct file/dir
    if isinstance(role_spec, dict):
        pid_str = role_spec.get("id")
        if not isinstance(pid_str, str):
            raise ValueError(f"role_spec dict must contain string 'id'. Got: {role_spec}")
        # Highest priority: explicit file path
        if "file" in role_spec and role_spec["file"]:
            inst = Path(role_spec["file"])
            return pid_str, inst
        # Next: explicit dir that contains instruction.md
        if "dir" in role_spec and role_spec["dir"]:
            d = Path(role_spec["dir"])
            inst = d / "instruction.md" if d.is_dir() else d
            return pid_str, inst
    elif isinstance(role_spec, str):
        pid_str = role_spec
    else:
        raise TypeError(f"Unsupported role spec type: {type(role_spec)}")

    # 2) build from pid_str
    family, role, ver = _parse_pid(pid_str)
    inst = root / "prompts" / family / role / _v(ver) / "instruction.md"
    return pid_str, inst

def _resolve_app_prompt(root: Path, pid: str) -> Path:
    # pid like: "assistant_core/fa_IR@1.0.0"
    name, locale_ver = pid.split("/", 1)
    locale, ver = locale_ver.split("@")
    return root / "app_prompts" / name / locale / _v(ver) / "system.md"

def _resolve_dataset(root: Path, dsid: str) -> Path:
    # dsid like: "generic/qa-core@1.3.0"
    domain, name_ver = dsid.split("/",1)
    name, ver = name_ver.split("@")
    return root / "datasets" / domain / name / _v(ver) / "data.jsonl"

def _resolve_retrieval_bundle(root: Path, rid: str) -> Path:
    # rid like: "arctic-l-v2.0-hnsw@1.2.0"
    name, ver = rid.split("@")
    return root / "retrieval" / name / _v(ver) / "bundle.yaml"

def _resolve_judge(root: Path, kind: str, jid: str) -> Path:
    # kind: llm | embeddings; jid: "<name>@<ver>"
    print(jid)
    print(kind)

    name, ver = jid.split("@")
    return root / "judges" / kind / name / _v(ver) / "config.yaml"

def resolve_suite(suite_path: str) -> dict:
    root = Path(suite_path).parent.parent  # repo root (assuming config/suite.yaml)
    suite = read_yaml(suite_path)

    # datasets
    ds_entries = []
    for ds in suite.get("datasets", []):
        dsid = ds["id"] if isinstance(ds, dict) else ds
        p = _resolve_dataset(root, dsid)
        ds_entries.append({
            "id": dsid, "path": str(p), "sha256": sha256_file(str(p)),
            "size": count_jsonl(str(p)), "schema": "qa_v1"
        })

    # retrieval (optional)
    retrieval_entry = {}
    if "retrieval" in suite:
        rb_file = (root / suite["retrieval"]["bundle"]).resolve()
        retrieval_entry = {
            "bundle": {
                "id": suite["retrieval"]["bundle"],
                "file": str(rb_file),
                "sha256": sha256_file(rb_file)
            },
            "dataset": suite["retrieval"]["dataset"],  # pass through: name/path
            "eval": suite["retrieval"].get("eval", {}),
            "couple_into_generation": bool(suite["retrieval"].get("couple_into_generation", False)),
        }

    # generation (optional)
    generation_entry = suite.get("generation", {})


    # judges
    llm_id = suite["judges"]["llm"]
    emb_id = suite["judges"]["embeddings"]
    llm_path = _resolve_judge(root, "llm", llm_id)
    emb_path = _resolve_judge(root, "embeddings", emb_id)

    # system prompts
    sys_prompts = {}
    for name, role_spec in suite.get("system_prompts", {}).items():
        sp = _resolve_app_prompt(root, role_spec)
        sys_prompts[name] = {"id": role_spec, "file": str(sp), "sha256": sha256_file(str(sp))}

    # metric prompts
    prom_out = {}
    for metric, roles in suite.get("prompts", {}).items():
        print(f"promets item: metric: {metric} , roles: {roles}")
        prom_out[metric] = {}
        prom_out[metric]['base_dir']=  f'{root / "prompts" / metric}'
        for role, role_spec in roles.items():
            print(f"promets item: role: {role} , role_spec {role_spec} , root: {root}")
            mp = _resolve_prompts(root, role_spec)
            prom_out[metric][role] = {"id": role_spec, "file": str(mp)}#, "sha256": sha256_file(str(mp))}

    # metrics impl specs (we keep impl_id verbatim; code provenance via git commit)
    metrics = []
    for m in suite.get("metrics", []):
        metrics.append({
            "name": m["name"],
            "impl_id": m["impl"],
            "params": m.get("params", {}),
            "datasets": m.get("datasets", [d if isinstance(d,str) else d["id"] for d in suite["datasets"]]),
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
        "retrieval": retrieval_entry,
        "generation": generation_entry,
        "judges": {
            "llm": {"id": llm_id, "file": str(llm_path), "sha256": sha256_file(str(llm_path))},
            "embeddings": {"id": emb_id, "file": str(emb_path), "sha256": sha256_file(str(emb_path))},
        },
        "system_prompts": sys_prompts,
        "prompts": prom_out,
        "metrics": metrics,
        "reporting": suite.get("reporting", {}),
        "output": suite.get("output", {"base_dir":"runs"}),
    }

    lock = {
        "manifest_run_id": manifest["run"]["id"],
        "python": os.popen("python -V").read().strip(),
        "pip_freeze": os.popen("pip freeze").read().splitlines(),
        "env": {
            "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED",""),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES",""),
        }
    }
    return {"manifest": manifest, "lock": lock}

def write_manifest_and_lock(out_dir: str, suite_path: str):
    from .files import ensure_dir, write_yaml
    ensure_dir(out_dir)
    r = resolve_suite(suite_path)
    write_yaml(os.path.join(out_dir, "manifest.yaml"), r["manifest"])
    write_yaml(os.path.join(out_dir, "lock.yaml"), r["lock"])
    return r["manifest"], r["lock"]