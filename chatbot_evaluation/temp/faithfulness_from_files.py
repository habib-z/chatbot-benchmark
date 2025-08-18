from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
import typing as t
from pydantic import BaseModel, Field, ValidationError

from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics.base import MetricType, MetricWithLLM, SingleTurnMetric
from ragas.prompt import PydanticPrompt

# ---------- I/O models: identical to RAGAS defaults ----------
class StatementGeneratorInput(BaseModel):
    question: str = Field(description="The question to answer")
    answer: str = Field(description="The answer to the question")

class StatementGeneratorOutput(BaseModel):
    statements: t.List[str] = Field(description="The generated statements")

class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")

class NLIStatementOutput(BaseModel):
    statements: t.List[StatementFaithfulnessAnswer]

class NLIStatementInput(BaseModel):
    context: str = Field(..., description="The context of the question")
    statements: t.List[str] = Field(..., description="The statements to judge")

# ---------- File-backed PydanticPrompt ----------
class FileBackedPrompt(PydanticPrompt[t.Any, t.Any]):
    def __init__(self, instruction_text: str, input_model, output_model, examples):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = input_model
        self.output_model = output_model
        self.examples = examples  # list[(InModel, OutModel)]

def _must_read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required prompt file missing: {path}")
    return path.read_text(encoding="utf-8")

def _must_read_examples(path: Path, InModel, OutModel):
    if not path.exists():
        raise FileNotFoundError(f"Required examples file missing: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            obj = json.loads(line)
            try:
                inp = InModel(**obj["input"])
                o = OutModel(**obj["output"])
            except (KeyError, ValidationError) as e:
                raise ValueError(f"Bad example at {path}:{i}: {e}")
            out.append((inp, o))
    if not out:
        raise ValueError(f"No examples found in {path}")
    return out

def _join_context(ctx):
    if isinstance(ctx, list):
        return "\n".join([c for c in ctx if isinstance(c, str)])
    if isinstance(ctx, str):
        return ctx
    return ""

@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    """
    File-driven Faithfulness metric:
      - NO fallbacks: errors if prompt files are missing.
      - Same columns/outputs as RAGAS's default Faithfulness (score in [0,1]).
      - Saves underlying computation (statements + NLI verdicts) to sample._ragas_details['faithfulness'].
    """
    name: str = "faithfulness"
    base_dir: str = field(default="benchmarks/faithfulness/v1")

    statement_generator_prompt: PydanticPrompt = field(init=False)
    nli_statements_prompt: PydanticPrompt = field(init=False)

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def __post_init__(self):
        base = Path(self.base_dir)

        sg_instr = _must_read(base / "statement_generator.instruction.md")
        sg_ex = _must_read_examples(
            base / "statement_generator.examples.jsonl",
            StatementGeneratorInput, StatementGeneratorOutput
        )
        self.statement_generator_prompt = FileBackedPrompt(
            instruction_text=sg_instr,
            input_model=StatementGeneratorInput,
            output_model=StatementGeneratorOutput,
            examples=sg_ex,
        )

        nli_instr = _must_read(base / "nli_judge.instruction.md")
        nli_ex = _must_read_examples(
            base / "nli_judge.examples.jsonl",
            NLIStatementInput, NLIStatementOutput
        )
        self.nli_statements_prompt = FileBackedPrompt(
            instruction_text=nli_instr,
            input_model=NLIStatementInput,
            output_model=NLIStatementOutput,
            examples=nli_ex,
        )

    async def _create_statements(self, row: dict, callbacks):
        assert self.llm is not None, "LLM must be set"
        inp = StatementGeneratorInput(question=row["user_input"], answer=row["response"])
        out: StatementGeneratorOutput = await self.statement_generator_prompt.generate(
            llm=self.llm, data=inp, callbacks=callbacks
        )
        return out

    async def _create_verdicts(self, row: dict, statements: list[str], callbacks):
        assert self.llm is not None, "LLM must be set"
        nli_in = NLIStatementInput(
            context=_join_context(row.get("retrieved_contexts", [])),
            statements=statements
        )
        out: NLIStatementOutput = await self.nli_statements_prompt.generate(
            data=nli_in, llm=self.llm, callbacks=callbacks
        )
        return out

    def _compute_score(self, nli_out: NLIStatementOutput) -> float:
        n = len(nli_out.statements)
        if n == 0:
            return float("nan")
        supported = sum(1 for s in nli_out.statements if int(s.verdict) == 1)
        return supported / n

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        row = sample.to_dict()
        # 1) extract claims
        sg_out = await self._create_statements(row, callbacks)

        statements = list(sg_out.statements or [])
        print(f"statements: len:{len(statements)}")
        print(statements)
        if not statements:
            details = {"statements": [], "verdicts": [], "score": float("nan")}
            sample.__dict__.setdefault("_ragas_details", {})["faithfulness"] = details
            return float("nan")

        # 2) judge claims vs context
        nli_out = await self._create_verdicts(row, statements, callbacks)
        score = self._compute_score(nli_out)

        # 3) persist internals
        details = {
            "question": row.get("user_input"),
            "answer": row.get("response"),
            "contexts": row.get("retrieved_contexts", []),
            "statements": statements,
            "verdicts": [s.model_dump() for s in nli_out.statements],
            "score": score,
        }
        sample.__dict__.setdefault("_ragas_details", {})["faithfulness"] = details
        return float(score)