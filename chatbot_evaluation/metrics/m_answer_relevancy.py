# chatbot_evaluation/metrics/answer_relevancy_filebacked.py
from __future__ import annotations
import asyncio, json
import re
from dataclasses import dataclass, field
from pathlib import Path
import typing as t

import numpy as np
from pydantic import BaseModel
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricType, MetricOutputType, MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric
)
from ragas.prompt import PydanticPrompt

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

# -------- IO models --------
class ResponseRelevanceInput(BaseModel):
    response: str

class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int  # 1 -> non-committal / generic

class _FileBackedPrompt(PydanticPrompt[t.Any, t.Any]):
    def __init__(self, instruction_text: str, input_model, output_model, examples):
        super().__init__()
        self.instruction = instruction_text
        self.input_model = input_model
        self.output_model = output_model
        self.examples = examples

# -------- file utils --------
def _must_read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"Missing required prompt file: {p}")
    return p.read_text(encoding="utf-8").strip()

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

# -------- metric --------
@dataclass
class AnswerRelevancyFileBacked(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Answer relevancy:
      - LLM generates K (strictness) proxy questions from the response
      - Embedding sim(question, generated_questions), mean cosine
      - If any output is non-committal -> score := 0
      - File-backed prompt; returns {'score','details'} from single_turn_eval()
    """
    name: str = "answer_relevancy"
    base_dir: str = field(default="benchmarks/answer_relevancy/v1")
    strictness: int = 3  # how many generations
    spec_param: dict | None = field(default=None)

    question_generation: PydanticPrompt = field(init=False)

    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response"}
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    # ---- prompt loading ----
    def __post_init__(self):
        base = Path(self.base_dir)
        sp = self.spec_param or {}
        rr_ver_pref = sp.get("response_relevance_version")
        rr_root = base /  "response_relevance"
        rr_ver=_pick_version(rr_root, rr_ver_pref)
        instruction = _must_read_text(rr_root/rr_ver / "instruction.md")
        examples = _must_read_examples_jsonl(
            rr_root/rr_ver / "examples.jsonl",
            ResponseRelevanceInput,
            ResponseRelevanceOutput,
        )
        self.question_generation = _FileBackedPrompt(
            instruction_text=instruction,
            input_model=ResponseRelevanceInput,
            output_model=ResponseRelevanceOutput,
            examples=examples,
        )

    # ---- calc ----
    def _calculate_similarity(self, question: str, generated_questions: list[str]) -> list[float]:
        assert self.embeddings is not None, "answer_relevancy needs embeddings"
        if not generated_questions:
            return []
        qv = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        dvs = np.asarray(self.embeddings.embed_documents(generated_questions)).reshape(len(generated_questions), -1)
        denom = (np.linalg.norm(dvs, axis=1) * np.linalg.norm(qv, axis=1))
        denom = np.where(denom == 0, 1e-12, denom)
        sims = (np.dot(dvs, qv.T).reshape(-1,) / denom)
        return [float(x) for x in sims]

    def _score_and_details(
        self,
        question: str,
        response: str,
        outputs: t.Sequence[ResponseRelevanceOutput],
    ) -> tuple[float, dict]:
        gen_questions = [o.question.strip() for o in outputs]
        noncommittals = [int(o.noncommittal) for o in outputs if hasattr(o, "noncommittal")]
        any_non = any(noncommittals) if noncommittals else False

        if not any(q for q in gen_questions):
            cosines: list[float] = []
            mean_cos, score = float("nan"), float("nan")
        else:
            cosines = self._calculate_similarity(question, gen_questions)
            mean_cos = float(np.mean(cosines)) if cosines else float("nan")
            score = 0.0 if any_non else mean_cos

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
                "any_noncommittal": bool(any_non),
            },
            "score": float(score) if score == score else score,  # keep NaN if NaN
        }
        return float(score) if score == score else score, details

    # ---- async core used by ragas ----
    async def _ascore(self, row: dict, callbacks: Callbacks) -> float:
        assert self.llm is not None, "llm is not set"
        prompt_input = ResponseRelevanceInput(response=row["response"])
        tasks = [self.question_generation.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
                 for _ in range(self.strictness)]
        outs: list[ResponseRelevanceOutput] = await asyncio.gather(*tasks)
        score, _ = self._score_and_details(row["user_input"], row["response"], outs)
        return float(score) if score == score else score

    # ---- nice API for your runner ----
    async def _eval_async(self, sample: SingleTurnSample, callbacks) -> dict:
        row = sample.to_dict()
        assert row.get("user_input") is not None and row.get("response") is not None, "need user_input/response"
        assert self.llm is not None, "llm is not set"
        prompt_input = ResponseRelevanceInput(response=row["response"])
        tasks = [self.question_generation.generate(data=prompt_input, llm=self.llm, callbacks=callbacks)
                 for _ in range(self.strictness)]
        outs: list[ResponseRelevanceOutput] = await asyncio.gather(*tasks)
        score, details = self._score_and_details(row["user_input"], row["response"], outs)

        # back-compat: attach to sample
        bucket = sample.__dict__.setdefault("_ragas_details", {})
        bucket[self.name] = details

        return {"score": float(score) if score == score else score, "details": details}

    def single_turn_eval(self, sample: SingleTurnSample) -> dict:
        import asyncio
        async def _run(): return await self._eval_async(sample, callbacks=None)
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