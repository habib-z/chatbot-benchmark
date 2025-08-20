from __future__ import annotations

import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass, field
from pathlib import Path

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

logger = logging.getLogger(__name__)


# ----------------------------
# IO models (IDENTICAL to RAGAS)
# ----------------------------
class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int


class ResponseRelevanceInput(BaseModel):
    response: str


# ----------------------------
# RAGAS default prompt (kept for compatibility)
# ----------------------------
class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = (
        'Generate a question for the given answer and Identify if answer is noncommittal. '
        'Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. '
        'A noncommittal answer is one that is evasive, vague, or ambiguous. For example, '
        '"I don\'t know" or "I\'m not sure" are noncommittal answers'
    )
    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="Albert Einstein was born in Germany.",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
            ),
        ),
        (
            ResponseRelevanceInput(
                response="I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. ",
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
            ),
        ),
    ]


# ----------------------------
# File-backed prompt wrapper
# ----------------------------
class _FileBackedPrompt(PydanticPrompt[t.Any, t.Any]):
    def __init__(self, instruction_text: str, input_model, output_model, examples):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = input_model
        self.output_model = output_model
        self.examples = examples


def _must_read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def _must_read_examples_jsonl(
    path: Path, InModel, OutModel
) -> list[tuple[BaseModel, BaseModel]]:
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


# ----------------------------
# ResponseRelevancy (IDENTICAL logic/signature to RAGAS)
# ----------------------------
@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    RAGAS original: scores relevancy of answer to the given question by
    generating K questions from the answer and comparing to the original
    question via embedding cosine; zeroes score if any generation is noncommittal.
    """

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    # In RAGAS this is instantiated from ResponseRelevancePrompt().
    # We keep the type and default here; our subclass will replace it with file-backed.
    question_generation: PydanticPrompt = field(default_factory=ResponseRelevancePrompt)
    strictness: int = 3

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert (
            self.embeddings is not None
        ), f"Error: '{self.name}' requires embeddings to be set."
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (np.dot(gen_question_vec, question_vec.T).reshape(-1,) / np.where(norm == 0, 1e-12, norm))

    def _calculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> float:
        question = row["user_input"]
        gen_questions = [answer.question for answer in answers]
        committal = np.any([answer.noncommittal for answer in answers])
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = float(cosine_sim.mean()) * int(not committal)

        return float(score)

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        return await self._ascore(row, callbacks)

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(response=row["response"])
        tasks = [
            self.question_generation.generate(
                data=prompt_input,
                llm=self.llm,
                callbacks=callbacks,
            )
            for _ in range(self.strictness)
        ]
        responses = await asyncio.gather(*tasks)

        return self._calculate_score(responses, row)


# ----------------------------
# AnswerRelevancy: file-backed prompts + full detail capture
# ----------------------------
@dataclass
class AnswerRelevancy(ResponseRelevancy):
    """
    Drop-in replacement:
    - Loads instruction/examples from files (no fallback).
    - Persists internals under sample._ragas_details['answer_relevancy'].
    - Scoring, signatures, and name remain identical to RAGAS.
    """
    base_dir: str = field(default="../benchmarks/answer_relevancy/v1")

    def __post_init__(self):
        base = Path(self.base_dir)
        instruction = _must_read_text(base / "response_relevance.instruction.md")
        examples = _must_read_examples_jsonl(
            base / "response_relevance.examples.jsonl",
            ResponseRelevanceInput,
            ResponseRelevanceOutput,
        )
        # Replace the RAGAS default prompt with our file-backed prompt
        self.question_generation = _FileBackedPrompt(
            instruction_text=instruction,
            input_model=ResponseRelevanceInput,
            output_model=ResponseRelevanceOutput,
            examples=examples,
        )

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        """
        Identical compute as base, but we ALSO persist all judge outputs & sims.
        """
        row = sample.to_dict()
        assert self.llm is not None, "LLM is not set"
        assert "user_input" in row and "response" in row, "Missing user_input/response"

        # 1) generate K questions from the answer (parallel; same as base)
        prompt_input = ResponseRelevanceInput(response=row["response"])
        tasks = [
            self.question_generation.generate(
                data=prompt_input, llm=self.llm, callbacks=callbacks
            )
            for _ in range(self.strictness)
        ]
        outputs: list[ResponseRelevanceOutput] = await asyncio.gather(*tasks)

        # 2) compute the SAME score as base class
        question = row["user_input"]
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

        # 3) persist ALL raw judge outputs + sims
        details = {
            "user_input": question,
            "response": row["response"],
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

        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket["answer_relevancy"] = details
        bucket["answer_relevancy_details"] = details  # alias

        return float(score)


# Keep the variable name that RAGAS exports
answer_relevancy = AnswerRelevancy()
