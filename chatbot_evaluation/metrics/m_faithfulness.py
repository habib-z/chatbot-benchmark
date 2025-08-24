# faithfulness_filebacked.py
from __future__ import annotations
import json, re, os, hashlib
from pathlib import Path
from dataclasses import dataclass, field
import typing as t

from pydantic import BaseModel
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType, MetricOutputType
from ragas.prompt import PydanticPrompt
from ragas.metrics import Faithfulness as RagasFaithfulness  # base class

# ---- small file utils ----
def _must_read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required prompt file: {path}")
    return path.read_text(encoding="utf-8")

def _must_read_examples(path: Path, InModel, OutModel) -> list[tuple[BaseModel, BaseModel]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required examples file: {path}")
    out: list[tuple[BaseModel, BaseModel]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inp = InModel(**obj["input"])
            outp = OutModel(**obj["output"])
            out.append((inp, outp))
    if not out:
        raise ValueError(f"No examples found in {path}")
    return out

class _FileBackedPrompt(PydanticPrompt[t.Any, t.Any]):
    def __init__(self, instruction_text: str, input_model, output_model, examples):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = input_model
        self.output_model = output_model
        self.examples = examples

def _sha_key(row: dict, qid: t.Optional[str]) -> str:
    if qid is not None:
        return str(qid)
    m = hashlib.sha256()
    m.update((row.get("user_input","") + "\x1f" + row.get("response","")).encode("utf-8"))
    for ctx in (row.get("retrieved_contexts") or []):
        m.update(b"\x1e"); m.update((ctx or "").encode("utf-8"))
    return m.hexdigest()

def _pick_version(dirpath: Path, preferred: str | None) -> str:
    """
    Pick a version folder like v1, v1.2.0 under dirpath.
    If preferred provided and exists -> use it.
    Else choose lexicographically max with 'v' prefix using a numeric-aware sort on dotted parts.
    """
    if preferred:
        candidate = dirpath / preferred
        if candidate.exists() and candidate.is_dir():
            return preferred
    vers = []
    for child in dirpath.iterdir():
        if child.is_dir() and child.name.startswith("v"):
            # extract numeric tuple (v1.2.3) -> (1,2,3)
            nums = tuple(int(x) for x in re.findall(r"\d+", child.name))
            vers.append((nums, child.name))
    if not vers:
        raise FileNotFoundError(f"No version folders under {dirpath}")
    # pick max by numeric tuple
    vers.sort()
    return vers[-1][1]

@dataclass
class FaithfulnessFileBacked(RagasFaithfulness):
    """
    Exact-behavior Faithfulness with file-backed prompts and full detail capture.

    - SAME I/O + scoring as base
    - Version-aware prompt loading (no hard-coded v1)
    - Clean API: single_turn_eval() returns {'score', 'details'}
    - Back-compat: still writes sample._ragas_details[self.name]
    """
    base_dir: str = field(default="benchmarks/faithfulness")
    # optional per-role versions. If None, auto-pick latest v*
    spec_param: dict | None = field(default=None)

    # (kept for compatibility, but you won't need it with single_turn_eval)
    _details_cache: dict = field(default_factory=dict)

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    # ---- prompt loading with versions ----
    def __post_init__(self):
        base = Path(self.base_dir)

        # import I/O models from base prompts to ensure identity
        SG_In  = self.statement_generator_prompt.input_model
        SG_Out = self.statement_generator_prompt.output_model
        NLI_In = self.nli_statements_prompt.input_model
        NLI_Out= self.nli_statements_prompt.output_model

        # versions (from spec_param or auto)
        sp = self.spec_param or {}
        sg_ver  = sp.get("statement_generator_version")  # e.g., "v1" or "v1.1.0"
        nli_ver = sp.get("nli_judge_version")            # e.g., "v2"

        sg_dir  = base / "statement_generator"
        nli_dir = base / "nli_judge"

        print(f"sg_ver: {sg_ver},\n nli_ver: {nli_ver},\n sg_dir: {sg_dir},\n nli_dir: {nli_dir}")

        sg_ver  = _pick_version(sg_dir, sg_ver)
        nli_ver = _pick_version(nli_dir, nli_ver)

        sg_instruction = _must_read(sg_dir / sg_ver / "instruction.md")
        sg_examples    = _must_read_examples(sg_dir / sg_ver / "examples.jsonl", SG_In, SG_Out)
        nli_instruction = _must_read(nli_dir / nli_ver / "instruction.md")
        nli_examples    = _must_read_examples(nli_dir / nli_ver / "examples.jsonl", NLI_In, NLI_Out)

        # replace prompts with file-backed versions
        self.statement_generator_prompt = _FileBackedPrompt(
            instruction_text=sg_instruction,
            input_model=SG_In,
            output_model=SG_Out,
            examples=sg_examples,
        )
        self.nli_statements_prompt = _FileBackedPrompt(
            instruction_text=nli_instruction,
            input_model=NLI_In,
            output_model=NLI_Out,
            examples=nli_examples,
        )

        # keep for details
        self._sg_version = sg_ver
        self._nli_version = nli_ver

    # ---- base wrappers to capture exact LLM I/O ----
    async def _create_statements(self, row: dict, callbacks):
        assert self.llm is not None, "llm is not set"
        text, question = row["response"], row["user_input"]
        prompt_input = self.statement_generator_prompt.input_model(question=question, answer=text)
        self.__dict__["_last_sg_input"] = prompt_input.model_dump()
        statements = await self.statement_generator_prompt.generate(llm=self.llm, data=prompt_input, callbacks=callbacks)
        self.__dict__["_last_sg_output"] = statements.model_dump()
        return statements

    async def _create_verdicts(self, row: dict, statements: list[str], callbacks):
        assert self.llm is not None, "llm must be set to compute score"
        contexts_str: str = "\n".join(row["retrieved_contexts"])
        nli_in = self.nli_statements_prompt.input_model(context=contexts_str, statements=statements)
        self.__dict__["_last_nli_input"] = nli_in.model_dump()
        verdicts = await self.nli_statements_prompt.generate(data=nli_in, llm=self.llm, callbacks=callbacks)
        self.__dict__["_last_nli_output"] = verdicts.model_dump()
        return verdicts

    # ---- clean API you will use from the runner ----
    async def _eval_async(self, sample: SingleTurnSample, callbacks) -> dict:
        """
        Returns {'score': float, 'details': dict} without relying on buckets/caches.
        Also writes details to sample._ragas_details[self.name] for compatibility.
        """
        row = sample.to_dict()

        # 1) statements
        sg_out = await self._create_statements(row, callbacks)
        statements = list(sg_out.statements or [])
        if not statements:
            details = {
                "meta": {
                    "metric": self.name,
                    "sg_version": getattr(self, "_sg_version", None),
                    "nli_version": getattr(self, "_nli_version", None),
                    "qid": getattr(sample, "id", None) or _sha_key(row, None),
                },
                "io": {
                    "question": row.get("user_input"),
                    "answer": row.get("response"),
                    "contexts": row.get("retrieved_contexts", []),
                },
                "generator": {
                    "instruction": self.statement_generator_prompt.instruction,
                    "input": self.__dict__.get("_last_sg_input"),
                    "output": self.__dict__.get("_last_sg_output"),
                },
                "nli": None,
                "artifacts": {"statements": [], "verdicts": []},
                "score": float("nan"),
            }
            # attach for back-compat
            bucket = sample.__dict__.setdefault("_ragas_details", {})
            bucket[self.name] = details
            return {"score": float("nan"), "details": details}

        # 2) verdicts
        nli_out = await self._create_verdicts(row, statements, callbacks)
        supported = sum(1 for a in nli_out.statements if int(a.verdict) == 1)
        score = supported / len(nli_out.statements) if nli_out.statements else float("nan")

        details = {
            "meta": {
                "metric": self.name,
                "sg_version": getattr(self, "_sg_version", None),
                "nli_version": getattr(self, "_nli_version", None),
                "qid": getattr(sample, "id", None) or _sha_key(row, None),
            },
            "io": {
                "question": row.get("user_input"),
                "answer": row.get("response"),
                "contexts": row.get("retrieved_contexts", []),
            },
            "generator": {
                "instruction": self.statement_generator_prompt.instruction,
                "input": self.__dict__.get("_last_sg_input"),
                "output": self.__dict__.get("_last_sg_output"),
            },
            "nli": {
                "instruction": self.nli_statements_prompt.instruction,
                "input": self.__dict__.get("_last_nli_input"),
                "output": self.__dict__.get("_last_nli_output"),
            },
            "artifacts": {
                "statements": statements,
                "verdicts": [s.model_dump() for s in nli_out.statements],
            },
            "score": float(score),
        }

        # attach for back-compat
        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket[self.name] = details
        return {"score": float(score), "details": details}

    # public sync helpers
    def single_turn_eval(self, sample: SingleTurnSample) -> dict:
        import asyncio
        async def _run():
            return await self._eval_async(sample, callbacks=None)
        try:
            return asyncio.run(_run())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(_run())

    def single_turn_score(self, sample: SingleTurnSample) -> float:
        return self.single_turn_eval(sample)["score"]