from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
import typing as t

from pydantic import BaseModel
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import MetricType
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


@dataclass
class FaithfulnessFileBacked(RagasFaithfulness):
    """
    Exact-behavior Faithfulness with file-backed prompts and full detail capture.

    - Uses the SAME I/O models and scoring as the built-in class.
    - No fallback: raises if prompt files are missing.
    - Persists internals under both ['faithfulness'] and ['faithfulness_details'] on the sample,
      and also into a metric-level cache: self._details_cache[key].
    """
    base_dir: str = field(default="benchmarks/faithfulness/v0")

    # metric-level details cache (in case ragas copies the sample internally)
    _details_cache: dict = field(default_factory=dict)

    # keep required columns identical to base
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )

    def __post_init__(self):
        # load instruction/examples; error if missing
        base = Path(self.base_dir)

        # import input/output models from the base class' prompts to ensure identity
        SG_In = self.statement_generator_prompt.input_model
        SG_Out = self.statement_generator_prompt.output_model
        NLI_In = self.nli_statements_prompt.input_model
        NLI_Out = self.nli_statements_prompt.output_model

        sg_instruction = _must_read(base / "statement_generator.instruction.md")
        sg_examples = _must_read_examples(base / "statement_generator.examples.jsonl", SG_In, SG_Out)
        nli_instruction = _must_read(base / "nli_judge.instruction.md")
        nli_examples = _must_read_examples(base / "nli_judge.examples.jsonl", NLI_In, NLI_Out)

        # replace prompts with file-backed versions (same I/O models)
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

    # ---- wrap base internals to capture exact LLM I/O ----
    async def _create_statements(self, row: dict, callbacks):
        assert self.llm is not None, "llm is not set"
        text, question = row["response"], row["user_input"]
        prompt_input = self.statement_generator_prompt.input_model(
            question=question, answer=text
        )

        print("input")
        # record exact input sent to LLM
        self.__dict__["_last_sg_input"] = prompt_input.model_dump()
        statements = await self.statement_generator_prompt.generate(
            llm=self.llm, data=prompt_input, callbacks=callbacks
        )
        # record parsed output
        self.__dict__["_last_sg_output"] = statements.model_dump()
        print(statements)
        return statements

    async def _create_verdicts(self, row: dict, statements: list[str], callbacks):
        assert self.llm is not None, "llm must be set to compute score"
        contexts_str: str = "\n".join(row["retrieved_contexts"])
        nli_in = self.nli_statements_prompt.input_model(
            context=contexts_str, statements=statements
        )
        # record exact input sent to LLM
        self.__dict__["_last_nli_input"] = nli_in.model_dump()
        verdicts = await self.nli_statements_prompt.generate(
            data=nli_in, llm=self.llm, callbacks=callbacks
        )
        # record parsed output
        self.__dict__["_last_nli_output"] = verdicts.model_dump()
        return verdicts

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks) -> float:
        """
        Identical flow to base, BUT we also persist all internals to:
          - sample._ragas_details['faithfulness'] and ['faithfulness_details']
          - self._details_cache[key]
        """

        print("_single_turn_ascore")
        row = sample.to_dict()

        # 1) generate statements (uses file-backed statement_generator_prompt)
        sg_out = await self._create_statements(row, callbacks)
        print("sg_out", sg_out)
        statements = list(sg_out.statements or [])
        if not statements:
            details = {
                "question": row.get("user_input"),
                "answer": row.get("response"),
                "contexts": row.get("retrieved_contexts", []),
                "generator": {
                    "instruction": self.statement_generator_prompt.instruction,
                    "input": self.__dict__.get("_last_sg_input"),
                    "output": self.__dict__.get("_last_sg_output"),
                },
                "nli": None,
                "statements": [],
                "verdicts": [],
                "score": float("nan"),
            }
            bucket = sample.__dict__.setdefault("_ragas_details", {})
            bucket["faithfulness"] = details
            bucket["faithfulness_details"] = details  # alias for compatibility

            # also cache at metric level
            key = getattr(sample, "id", None)
            if key is None:
                key = (hash(row.get("user_input")), hash(row.get("response")))
            self._details_cache[key] = details

            return float("nan")

        # 2) NLI judge (uses file-backed nli_statements_prompt)
        nli_out = await self._create_verdicts(row, statements, callbacks)
        print("nil out")
        # 3) compute the SAME score as base class
        supported = sum(1 for a in nli_out.statements if int(a.verdict) == 1)
        score = supported / len(nli_out.statements) if nli_out.statements else float("nan")

        # 4) persist ALL raw judge outputs
        details = {
            "question": row.get("user_input"),
            "answer": row.get("response"),
            "contexts": row.get("retrieved_contexts", []),
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
            "statements": statements,                                  # from generator
            "verdicts": [s.model_dump() for s in nli_out.statements],  # from NLI judge
            "score": score,
        }
        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket["faithfulness"] = details
        bucket["faithfulness_details"] = details  # alias
        print("nil out 2")
        # also cache at metric level (handles internal copies)
        key = getattr(sample, "id", None)
        if key is None:
            key = (hash(row.get("user_input")), hash(row.get("response")))
        self._details_cache[key] = details

        return float(score)