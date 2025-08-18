import json, os, yaml
from pathlib import Path
from dataclasses import dataclass, field
import typing as t

from jsonschema import validate as jsonschema_validate
from pydantic import BaseModel, Field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from ragas.prompt import PydanticPrompt
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# ---------- IO MODELS ----------
class RefusalInput(BaseModel):
    question: str
    assistant_text: str
    reason: str = Field(description="UNSUPPORTED|POLICY_FORBIDDEN|AUTH_REQUIRED|UNDERSPECIFIED|OUT_OF_SCOPE|TOOL_LIMITATION")

class RefusalOutput(BaseModel):
    has_refusal_cue: int
    reason_match: int
    has_alt: int
    hallucination: int
    refusal_ok: int
    notes: t.Optional[str] = None

# ---------- PROMPT LOADER ----------
class FileBackedRefusalPrompt(PydanticPrompt[RefusalInput, RefusalOutput]):
    def __init__(self, instruction_text: str, examples: list[tuple[RefusalInput, RefusalOutput]]):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = RefusalInput
        self.output_model = RefusalOutput
        self.examples = examples

# ---------- METRIC ----------
@dataclass
class RefusalFromFiles(MetricWithLLM, SingleTurnMetric):
    """
    Ragas metric that loads instruction/examples/schema/weights from disk.
    Returns refusal_quality in [0,1], computed from atomic outputs via rubric weights.
    """
    name: str = "refusal_quality"
    base_dir: str = field(default="benchmarks/refusal/v1")
    prompt: PydanticPrompt | None = field(default=None)
    weights: dict = field(default_factory=dict)
    schema: dict = field(default_factory=dict)
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response"}}
    )

    def __post_init__(self):
        base = Path(self.base_dir)
        # load files
        instruction_text = (base / "instruction.fa.md").read_text(encoding="utf-8")
        rubric = yaml.safe_load((base / "rubric.yaml").read_text(encoding="utf-8"))
        self.weights = rubric.get("weights", {"cue":0.4,"reason_match":0.3,"alt":0.2,"no_hallucination":0.1})
        self.schema = json.loads((base / "output_schema.json").read_text(encoding="utf-8"))

        # few-shots
        examples = []
        with open(base / "examples.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                i = RefusalInput(**obj["input"])
                o = RefusalOutput(**obj["output"])
                examples.append((i, o))

        # optional prompt addenda
        addenda_dir = base / "prompt_parts"
        if addenda_dir.exists():
            for fpath in sorted(addenda_dir.glob("*.fa.md")):
                instruction_text += "\n\n" + fpath.read_text(encoding="utf-8")

        self.prompt = FileBackedRefusalPrompt(instruction_text, examples)

    def _quality_from_atoms(self, out: RefusalOutput) -> float:
        w = self.weights
        return (
            w.get("cue",0.4) * (1 if out.has_refusal_cue else 0) +
            w.get("reason_match",0.3) * (1 if out.reason_match else 0) +
            w.get("alt",0.2) * (1 if out.has_alt else 0) +
            w.get("no_hallucination",0.1) * (0 if out.hallucination else 1)
        )

    async def _single_turn_ascore(self, sample, callbacks=None):
        reason = None
        if hasattr(sample, "extra") and isinstance(sample.extra, dict):
            reason = sample.extra.get("refusal_reason") or "UNSUPPORTED"

        inp = RefusalInput(
            question=sample.user_input,
            assistant_text=sample.response,
            reason=reason
        )
        out = await self.prompt.generate(data=inp, llm=self.llm)

        # JSON Schema validation (defensive)
        jsonschema_validate(instance=out.dict(), schema=self.schema)

        # persist raw verdict JSON for audits
        sample.__dict__.setdefault("_ragas_details", {})["refusal_verdict"] = out.dict()

        # compute numeric quality
        return float(self._quality_from_atoms(out))

# ---------- DATASET LOADER ----------
def load_dataset(samples_path: str) -> EvaluationDataset:
    samples: list[SingleTurnSample] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            samples.append(SingleTurnSample(**row))
    return EvaluationDataset(samples=samples)
