# factual_correctness_filebacked.py
from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from dataclasses import dataclass, field
import typing as t
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from ragas.metrics._faithfulness import NLIStatementInput, NLIStatementPrompt
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.metrics.utils import fbeta_score
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks
    from ragas.dataset_schema import SingleTurnSample


# --------- Pydantic IO models ----------
class ClaimDecompositionInput(BaseModel):
    response: str = Field(..., title="Response")

class ClaimDecompositionOutput(BaseModel):
    claims: t.List[str] = Field(..., title="Decomposed Claims")


class FileBackedPrompt(PydanticPrompt[t.Any, t.Any]):
    def __init__(self, instruction_text: str, input_model, output_model, examples):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = input_model
        self.output_model = output_model
        self.examples = examples


# --------- File utils ----------
def _must_read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()

def _must_read_examples_jsonl(path: Path, InModel, OutModel) -> list[tuple[BaseModel, BaseModel]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required examples file: {path}")
    out: list[tuple[BaseModel, BaseModel]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
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

def _pick_version(dirpath: Path, preferred: str | None) -> str:
    """
    Pick a version folder like v1, v1.2.0 under dirpath.
    If preferred provided and exists -> use it.
    Else choose numerically max v*.
    """
    if preferred:
        candidate = dirpath / preferred
        if candidate.exists() and candidate.is_dir():
            return preferred
    vers = []
    for child in dirpath.iterdir():
        if child.is_dir() and child.name.startswith("v"):
            nums = tuple(int(x) for x in re.findall(r"\d+", child.name))
            vers.append((nums, child.name))
    if not vers:
        raise FileNotFoundError(f"No version folders under {dirpath}")
    vers.sort()
    return vers[-1][1]

def _sha_key_for_sample(row: dict, qid: t.Optional[str]) -> str:
    if qid is not None:
        return str(qid)
    m = hashlib.sha256()
    m.update((row.get("response","") + "\x1f" + row.get("reference","")).encode("utf-8"))
    return m.hexdigest()


# --------- Metric ----------
@dataclass
class FactualCorrectnessFileBacked(MetricWithLLM, SingleTurnMetric):
    """
    File-backed FactualCorrectness with full judge I/O capture.
    Output matches FaithfulnessFileBacked: single_turn_eval() -> {'score', 'details'}
    Still writes details to sample._ragas_details['factual_correctness'] for back-compat.
    """
    name: str = "factual_correctness"
    base_dir: str = field(default="benchmarks/factual_correctness")
    # optional per-role versions. If None, auto-pick latest v*
    spec_param: dict | None = field(default=None)

    # params (unchanged)
    mode: t.Literal["precision", "recall", "f1"] = "f1"
    beta: float = 1.0
    atomicity: t.Literal["low", "high"] = "low"
    coverage: t.Literal["low", "high"] = "low"
    language: str = "english"

    # prompts filled in __post_init__
    claim_decomposition_prompt: PydanticPrompt = field(init=False)
    nli_prompt: PydanticPrompt = field(init=False)

    # optional metric-level cache (kept for parity)
    _details_cache: dict = field(default_factory=dict)

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS

    def __post_init__(self):
        if not isinstance(self.beta, float):
            raise ValueError("Beta must be a float.")

        base = Path(self.base_dir)
        sp = self.spec_param or {}
        cd_ver_pref = sp.get("claim_decomposition_version")  # e.g., "v1" or "v1.1.0"
        nli_ver_pref = sp.get("nli_judge_version")          # e.g., "v2"

        cd_root = base / "claim_decomposition"
        nli_root = base / "nli_judge"

        cd_ver  = _pick_version(cd_root, cd_ver_pref)
        nli_ver = _pick_version(nli_root, nli_ver_pref)
        self._cd_version  = cd_ver
        self._nli_version = nli_ver

        # claim decomposition: instruction + examples by atomicity/coverage
        cd_instruction = _must_read_text(cd_root / cd_ver / "instruction.md")
        combo = f"{self.atomicity}_atomicity_{self.coverage}_coverage"
        file_map = {
            "low_atomicity_low_coverage":   cd_root / cd_ver / "low_atomicity_low_coverage.examples.jsonl",
            "low_atomicity_high_coverage":  cd_root / cd_ver / "low_atomicity_high_coverage.examples.jsonl",
            "high_atomicity_low_coverage":  cd_root / cd_ver / "high_atomicity_low_coverage.examples.jsonl",
            "high_atomicity_high_coverage": cd_root / cd_ver / "high_atomicity_high_coverage.examples.jsonl",
        }
        cd_path = file_map.get(combo)
        if cd_path is None:
            raise ValueError(f"Unsupported atomicity/coverage combination: {combo}")
        cd_examples = _must_read_examples_jsonl(cd_path, ClaimDecompositionInput, ClaimDecompositionOutput)
        self.claim_decomposition_prompt = FileBackedPrompt(
            instruction_text=cd_instruction,
            input_model=ClaimDecompositionInput,
            output_model=ClaimDecompositionOutput,
            examples=cd_examples,
        )

        # NLI judge: instruction + examples
        nli_instruction = _must_read_text(nli_root / nli_ver / "instruction.md")
        nli_examples = _must_read_examples_jsonl(
            nli_root / nli_ver / "examples.jsonl",
            NLIStatementInput,
            NLIStatementPrompt.output_model  # type: ignore
        )
        self.nli_prompt = FileBackedPrompt(
            instruction_text=nli_instruction,
            input_model=NLIStatementInput,
            output_model=NLIStatementPrompt.output_model,  # type: ignore
            examples=nli_examples,
        )

    # ---------- helpers (capture I/O exactly) ----------
    async def _decompose(self, text: str, which: str, callbacks: Callbacks) -> list[str]:
        assert self.llm is not None, "LLM must be set"
        inp = ClaimDecompositionInput(response=text)
        self.__dict__[f"_last_cd_{which}_input"] = inp.model_dump()
        out: ClaimDecompositionOutput = await self.claim_decomposition_prompt.generate(
            data=inp, llm=self.llm, callbacks=callbacks
        )
        self.__dict__[f"_last_cd_{which}_output"] = out.model_dump()
        return list(out.claims or [])

    async def _nli(self, context: str, statements: list[str], which: str, callbacks: Callbacks):
        assert self.llm is not None, "LLM must be set"
        inp = NLIStatementInput(context=context, statements=statements)
        self.__dict__[f"_last_nli_{which}_input"] = inp.model_dump()
        out = await self.nli_prompt.generate(data=inp, llm=self.llm, callbacks=callbacks)
        self.__dict__[f"_last_nli_{which}_output"] = out.model_dump()
        if out.statements:
            arr = np.array([bool(s.verdict) for s in out.statements], dtype=bool)
            raw = [s.model_dump() for s in out.statements]
        else:
            arr = np.array([], dtype=bool)
            raw = []
        return arr, raw

    # ---------- core async eval that returns {'score','details'} ----------
    async def _eval_async(self, sample: SingleTurnSample, callbacks) -> dict:
        assert self.llm is not None, "LLM must be set"
        assert sample.reference is not None, "Reference is not set"
        assert sample.response is not None, "Response is not set"

        row = sample.to_dict()
        reference = row["reference"]
        response  = row["response"]

        # A -> R
        resp_claims = await self._decompose(response, which="response", callbacks=callbacks)
        a2r_arr, a2r_raw = await self._nli(context=reference, statements=resp_claims, which="a2r", callbacks=callbacks)

        # R -> A (except precision-only mode)
        if self.mode != "precision":
            ref_claims = await self._decompose(reference, which="reference", callbacks=callbacks)
            r2a_arr, r2a_raw = await self._nli(context=response, statements=ref_claims, which="r2a", callbacks=callbacks)
        else:
            ref_claims, r2a_arr, r2a_raw = [], np.array([], dtype=bool), []

        # counts
        tp = int(a2r_arr.sum())
        fp = int((~a2r_arr).sum())
        fn = int((~r2a_arr).sum()) if self.mode != "precision" else 0

        # score
        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)
        score = float(np.round(score, 2))

        # details (aligned with FaithfulnessFileBacked structure)
        qid = getattr(sample, "id", None) or _sha_key_for_sample(row, None)
        details = {
            "meta": {
                "metric": self.name,
                "claim_decomposition_version": getattr(self, "_cd_version", None),
                "nli_version": getattr(self, "_nli_version", None),
                "mode": self.mode,
                "beta": self.beta,
                "atomicity": self.atomicity,
                "coverage": self.coverage,
                "qid": qid,
            },
            "io": {
                "response": response,
                "reference": reference,
            },
            "decomposition": {
                "instruction": self.claim_decomposition_prompt.instruction,
                "response": {
                    "input": self.__dict__.get("_last_cd_response_input"),
                    "output": self.__dict__.get("_last_cd_response_output"),
                },
                "reference": {
                    "input": self.__dict__.get("_last_cd_reference_input"),
                    "output": self.__dict__.get("_last_cd_reference_output"),
                } if self.mode != "precision" else None,
            },
            "nli": {
                "instruction": self.nli_prompt.instruction,
                "a2r": {
                    "input": self.__dict__.get("_last_nli_a2r_input"),
                    "output": self.__dict__.get("_last_nli_a2r_output"),
                },
                "r2a": {
                    "input": self.__dict__.get("_last_nli_r2a_input"),
                    "output": self.__dict__.get("_last_nli_r2a_output"),
                } if self.mode != "precision" else None,
            },
            "artifacts": {
                "response_claims": resp_claims,
                "reference_claims": ref_claims,
                "a2r_verdicts": a2r_raw,
                "r2a_verdicts": r2a_raw if self.mode != "precision" else [],
                "counts": {"tp": tp, "fp": fp, "fn": fn},
            },
            "score": score,
        }

        # back-compat attach
        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket[self.name] = details
        self._details_cache[qid] = details

        return {"score": score, "details": details}

    # ---------- public sync helpers ----------
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

    # keep ragas path working (returns float)
    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        res = await self._eval_async(sample, callbacks)
        return float(res["score"])