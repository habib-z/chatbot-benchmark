import json
from pathlib import Path
from dataclasses import dataclass, field
import typing as t

import yaml
from jsonschema import validate as jsonschema_validate
from pydantic import BaseModel, Field
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType
from ragas.prompt import PydanticPrompt
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# -------- IO MODELS --------
class FactualInput(BaseModel):
    question: str
    reference_text: str
    assistant_text: str

class FactualOutput(BaseModel):
    assistant_claims: list[str]
    reference_claims: list[str]
    correct_claims: list[str]
    missing_ref_claims: list[str]
    incorrect_claims: list[str]
    neutral_extra_claims: list[str]
    notes: t.Optional[str] = None

# -------- PROMPT (file-backed) --------
class FileBackedFactualPrompt(PydanticPrompt[FactualInput, FactualOutput]):
    def __init__(self, instruction_text: str, examples: list[tuple[FactualInput, FactualOutput]]):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = FactualInput
        self.output_model = FactualOutput
        self.examples = examples

# -------- METRIC --------
@dataclass
class FactualFromFiles(MetricWithLLM, SingleTurnMetric):
    """
    File-driven Factual Correctness. Returns F-beta as numeric score and stores raw verdict JSON.
    """
    name: str = "factual_fbeta"
    base_dir: str = field(default="benchmarks/factual_correctness/v1")
    prompt: PydanticPrompt | None = field(default=None)
    schema: dict = field(default_factory=dict)
    beta: float = 1.0
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {MetricType.SINGLE_TURN: {"user_input", "response", "reference"}}
    )

    def __post_init__(self):
        base = Path(self.base_dir)
        instruction = (base / "instruction.en.md").read_text(encoding="utf-8")
        self.schema = json.loads((base / "output_schema.json").read_text(encoding="utf-8"))
        rubric = yaml.safe_load((base / "rubric.yaml").read_text(encoding="utf-8"))
        self.beta = float(rubric.get("beta", 1.0))

        # load few-shots
        examples: list[tuple[FactualInput, FactualOutput]] = []
        with open(base / "examples.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                i = FactualInput(**obj["input"])
                o = FactualOutput(**obj["output"])
                examples.append((i, o))

        self.prompt = FileBackedFactualPrompt(instruction, examples)

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    @staticmethod
    def _normalize_list(xs: list[str]) -> list[str]:
        # minimal normalization; judge handles semantics already
        return [x.strip() for x in xs if isinstance(x, str) and x.strip()]

    def _compute_prf(self, out: FactualOutput) -> tuple[float,float,float]:
        # Precision = |correct| / |assistant_claims|
        # Recall    = |correct| / |reference_claims|
        ca = len(self._normalize_list(out.correct_claims))
        aa = len(self._normalize_list(out.assistant_claims))
        rr = len(self._normalize_list(out.reference_claims))

        precision = self._safe_div(ca, aa)
        recall = self._safe_div(ca, rr)
        b2 = self.beta * self.beta
        fbeta = (1 + b2) * self._safe_div(precision * recall, (b2 * precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, fbeta

    async def _single_turn_ascore(self, sample, callbacks=None):
        inp = FactualInput(
            question=sample.user_input,
            reference_text=sample.reference or "",
            assistant_text=sample.response or "",
        )
        out = await self.prompt.generate(data=inp, llm=self.llm)

        # Validate strict JSON shape
        jsonschema_validate(instance=out.dict(), schema=self.schema)

        # Persist raw verdict
        details = sample.__dict__.setdefault("_ragas_details", {})
        details["factual_verdict"] = out.dict()

        # Compute P/R/Fβ
        p, r, fbeta = self._compute_prf(out)
        details["factual_precision"] = p
        details["factual_recall"] = r
        details["factual_fbeta"] = fbeta

        # Return numeric score (Ragas aggregator will average over samples)
        return float(fbeta)

# -------- DATASET LOADER --------
def load_dataset(samples_path: str) -> EvaluationDataset:
    samples: list[SingleTurnSample] = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # Map 'reference' field in file -> Sample.reference
            samples.append(SingleTurnSample(
                user_input=row["user_input"],
                response=row["response"],
                reference=row.get("reference"),
                id=row.get("qid"),
                extra=row.get("extra")
            ))
    return EvaluationDataset(samples=samples)
