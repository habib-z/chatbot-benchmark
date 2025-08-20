from __future__ import annotations
import json
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


# --------- Pydantic IO models (same as your code) ----------
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


# --------- Metric ----------
@dataclass
class FactualCorrectnessFileBacked(MetricWithLLM, SingleTurnMetric):
    """
    File-backed FactualCorrectness with full judge I/O capture.
    Same params & scoring as your original implementation.
    """
    name: str = "factual_correctness"
    base_dir: str = field(default="benchmarks/factual_correctness/v1")

    # init params (identical surface to your class)
    mode: t.Literal["precision", "recall", "f1"] = "f1"
    beta: float = 1.0
    atomicity: t.Literal["low", "high"] = "low"
    coverage: t.Literal["low", "high"] = "low"
    language: str = "english"

    # prompts (filled in __post_init__)
    claim_decomposition_prompt: PydanticPrompt = field(init=False)
    nli_prompt: PydanticPrompt = field(init=False)

    # cache when ragas copies samples internally
    _details_cache: dict = field(default_factory=dict)

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"response", "reference"}}
    )
    output_type: t.Optional[MetricOutputType] = MetricOutputType.CONTINUOUS

    def __post_init__(self):
        if not isinstance(self.beta, float):
            raise ValueError("Beta must be a float.")

        base = Path(self.base_dir)

        # Claim decomposition instruction + examples selected by atomicity/coverage
        cd_instruction = _must_read_text(base / "claim_decomposition.instruction.md")
        combo = f"{self.atomicity}_atomicity_{self.coverage}_coverage"
        file_map = {
            "low_atomicity_low_coverage":  base / "claim_decomposition.low_atomicity_low_coverage.examples.jsonl",
            "low_atomicity_high_coverage": base / "claim_decomposition.low_atomicity_high_coverage.examples.jsonl",
            "high_atomicity_low_coverage": base / "claim_decomposition.high_atomicity_low_coverage.examples.jsonl",
            "high_atomicity_high_coverage":base / "claim_decomposition.high_atomicity_high_coverage.examples.jsonl",
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

        # NLI judge instruction + examples
        nli_instruction = _must_read_text(base / "nli_judge.instruction.md")
        nli_examples = _must_read_examples_jsonl(
            base / "nli_judge.examples.jsonl",
            NLIStatementInput,
            NLIStatementPrompt.output_model  # type: ignore
        )
        self.nli_prompt = FileBackedPrompt(
            instruction_text=nli_instruction,
            input_model=NLIStatementInput,
            output_model=NLIStatementPrompt.output_model,  # type: ignore
            examples=nli_examples,
        )

    # ---------- helpers to capture I/O ----------
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
        else:
            arr = np.array([], dtype=bool)
        return arr, out

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM must be set"
        assert sample.reference is not None, "Reference is not set"
        assert sample.response is not None, "Response is not set"

        reference = sample.reference
        response = sample.response

        # 1) Decompose and verify A->R
        resp_claims = await self._decompose(response, which="response", callbacks=callbacks)
        a2r_arr, a2r_out = await self._nli(context=reference, statements=resp_claims, which="a2r", callbacks=callbacks)

        # 2) If needed, decompose R and verify R->A
        if self.mode != "precision":
            ref_claims = await self._decompose(reference, which="reference", callbacks=callbacks)
            r2a_arr, r2a_out = await self._nli(context=response, statements=ref_claims, which="r2a", callbacks=callbacks)
        else:
            ref_claims = []
            r2a_arr = np.array([], dtype=bool)
            r2a_out = None

        # 3) TP/FP/FN
        tp = int(a2r_arr.sum())
        fp = int((~a2r_arr).sum())
        fn = int((~r2a_arr).sum()) if self.mode != "precision" else 0

        # 4) Score by mode
        if self.mode == "precision":
            score = tp / (tp + fp + 1e-8)
        elif self.mode == "recall":
            score = tp / (tp + fn + 1e-8)
        else:
            score = fbeta_score(tp, fp, fn, self.beta)

        score = float(np.round(score, 2))

        # 5) Persist ALL details
        details = {
            "response": response,
            "reference": reference,
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
            "counts": {"tp": tp, "fp": fp, "fn": fn},
            "mode": self.mode,
            "beta": self.beta,
            "atomicity": self.atomicity,
            "coverage": self.coverage,
            "score": score,
        }

        # Save on the sample
        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket["factual_correctness"] = details
        bucket["factual_correctness_details"] = details  # alias

        # Also save into metric cache
        key = getattr(sample, "id", None)
        if key is None:
            key = (hash(response), hash(reference))
        self._details_cache[key] = details

        return score
