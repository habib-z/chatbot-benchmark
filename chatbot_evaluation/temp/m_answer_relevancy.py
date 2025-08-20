from __future__ import annotations
import asyncio
import json
from pathlib import Path
from dataclasses import dataclass, field
import typing as t

import numpy as np
from pydantic import BaseModel
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


# --------- IO models (identical to your code) ---------
class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int

class ResponseRelevanceInput(BaseModel):
    response: str


class FileBackedPrompt(PydanticPrompt[t.Any, t.Any]):
    def __init__(self, instruction_text: str, input_model, output_model, examples):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = input_model
        self.output_model = output_model
        self.examples = examples


# --------- File utils ---------
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


# --------- Metric ---------
@dataclass
class AnswerRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    File-backed answer_relevancy with full I/O capture.
    Same semantics as your ResponseRelevancy / AnswerRelevancy.
    """
    name: str = "answer_relevancy"
    base_dir: str = field(default="benchmarks/answer_relevancy/v1")
    strictness: int = 3

    # prompt loaded in __post_init__
    question_generation: PydanticPrompt = field(init=False)

    # cache (in case ragas uses internal copies of the sample)
    _details_cache: dict = field(default_factory=dict)

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    def __post_init__(self):
        base = Path(self.base_dir)
        instruction = _must_read_text(base / "response_relevance.instruction.md")
        examples = _must_read_examples_jsonl(
            base / "response_relevance.examples.jsonl",
            ResponseRelevanceInput,
            ResponseRelevanceOutput,
        )
        self.question_generation = FileBackedPrompt(
            instruction_text=instruction,
            input_model=ResponseRelevanceInput,
            output_model=ResponseRelevanceOutput,
            examples=examples,
        )

    # identical math to your code
    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert self.embeddings is not None, f"Error: '{self.name}' requires embeddings to be set."
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_vecs = np.asarray(self.embeddings.embed_documents(generated_questions)).reshape(len(generated_questions), -1)
        denom = (np.linalg.norm(gen_vecs, axis=1) * np.linalg.norm(question_vec, axis=1))
        denom = np.where(denom == 0, 1e-12, denom)  # guard divide-by-zero
        sims = (np.dot(gen_vecs, question_vec.T).reshape(-1,) / denom)
        return sims

    def _score_and_details(
        self,
        question: str,
        response: str,
        outputs: t.Sequence[ResponseRelevanceOutput],
    ) -> tuple[float, dict]:
        gen_questions = [o.question for o in outputs]
        noncommittals = [int(o.noncommittal) for o in outputs]
        any_noncommittal = any(noncommittals)

        if all(q.strip() == "" for q in gen_questions):
            cosines = []
            mean_cos = float("nan")
            score = float("nan")
        else:
            cosines = [float(x) for x in self.calculate_similarity(question, gen_questions)]
            mean_cos = float(np.mean(cosines)) if len(cosines) else float("nan")
            score = mean_cos * (0 if any_noncommittal else 1)

        details = {
            "user_input": question,
            "response": response,
            "generation": {
                "instruction": self.question_generation.instruction,
                "outputs": [o.model_dump() for o in outputs],
                "strictness": self.strictness,
            },
            "similarity": {
                "generated_questions": gen_questions,
                "cosine_per_question": cosines,
                "mean_cosine": mean_cos,
            },
            "noncommittal": {
                "flags": noncommittals,
                "any_noncommittal": bool(any_noncommittal),
            },
            "score": float(score),
        }
        return float(score), details

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        """
        Run the metric AND persist details onto the sample (this is where we have 'sample').
        """
        row = sample.to_dict()
        assert self.llm is not None, "LLM is not set"
        assert "user_input" in row and "response" in row, "Missing user_input/response"

        # 1) generate K questions from the answer (parallel)
        prompt_input = ResponseRelevanceInput(response=row["response"])
        gen_tasks = [
            self.question_generation.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
            for _ in range(self.strictness)
        ]
        outputs: list[ResponseRelevanceOutput] = await asyncio.gather(*gen_tasks)

        # 2) score + build details
        score, details = self._score_and_details(
            question=row["user_input"],
            response=row["response"],
            outputs=outputs,
        )

        # 3) persist ALL details on the sample
        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket["answer_relevancy"] = details
        bucket["answer_relevancy_details"] = details  # alias

        # also cache at metric level (handles internal sample copies)
        key = getattr(sample, "id", None)
        if key is None:
            key = (hash(row["user_input"]), hash(row["response"]))
        self._details_cache[key] = details

        return float(score)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        Fallback path used by ragas internals; returns only the score.
        Does NOT attempt to write to 'sample'.
        """
        assert self.llm is not None, "LLM is not set"
        assert "user_input" in row and "response" in row, "Missing user_input/response"

        prompt_input = ResponseRelevanceInput(response=row["response"])
        gen_tasks = [
            self.question_generation.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
            for _ in range(self.strictness)
        ]
        outputs: list[ResponseRelevanceOutput] = await asyncio.gather(*gen_tasks)
        score, _ = self._score_and_details(
            question=row["user_input"],
            response=row["response"],
            outputs=outputs,
        )
        return float(score)
