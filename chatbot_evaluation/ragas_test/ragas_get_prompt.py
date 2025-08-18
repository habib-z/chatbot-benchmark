#!/usr/bin/env python3
"""
ragas_eval_template.py

- Instantiates multiple LLM-judge metrics with varied parameters
- Customizes each metric's prompt (instruction and/or examples)
- Evaluates a small example EvaluationDataset
- Saves the effective prompt(s) used by each metric to files: <metric_name>.txt
- Prints aggregate and per-sample scores
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

# ---- Ragas core ----
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

# ---- Metrics (LLM-judge + hybrid) ----
from ragas.metrics import (
    ResponseRelevancy,                 # aka "answer_relevancy" (LLM + embeddings)
    Faithfulness,                      # LLM-judge
    FaithfulnesswithHHEM,              # Faithfulness variant with batch/device knobs
    LLMContextPrecisionWithReference,  # LLM-judge
    LLMContextRecall,                  # LLM-judge
    FactualCorrectness,                # LLM-judge (claims, NLI)
    SemanticSimilarity,                # Embedding-only sanity signal (not LLM-judge)
    AspectCritic,                      # General-purpose LLM-judge (binary)
)

# ---- Judge LLM (OpenAI-compatible; works with vLLM base_url) ----
from ragas.llms import llm_factory  # lets you set model and base_url for OpenAI-compatible stacks


def build_example_dataset() -> EvaluationDataset:
    """
    Replace this with your real traces. SingleTurnSample fields:
    - user_input
    - response (model answer)
    - retrieved_contexts (list[str])
    - reference (ground-truth answer)  # used by some metrics
    """
    samples: List[SingleTurnSample] = [
        SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            response="The Eiffel Tower is in Paris, France.",
            retrieved_contexts=[
                "The Eiffel Tower is located in Paris on the Champ de Mars.",
                "It was completed in 1889."
            ],
            reference="The Eiffel Tower is located in Paris, France.",
        ),
        SingleTurnSample(
            user_input="Who developed the theory of relativity?",
            response="It was formulated by Albert Einstein.",
            retrieved_contexts=[
                "Albert Einstein proposed the special theory of relativity in 1905.",
                "General relativity was published in 1915."
            ],
            reference="Albert Einstein developed both special and general relativity.",
        ),
    ]
    return EvaluationDataset.from_list([s.dict() for s in samples])


def customize_metric_prompts(metric, *, instruction_suffix: str = "", example_patch: Any = None):
    """
    Generic helper: pull a metric's prompts, mutate instruction/examples, set back.
    - instruction_suffix: appended to instruction (e.g., domain constraints or output format)
    - example_patch: if provided, replaces `examples` on any prompt that has .examples
    """
    # prompts: Dict[str, Any] = metric.get_prompts()  # unified interface across metrics
    try:
        prompts: Dict[str, Any] = metric.get_prompts()
    except Exception as e:
        print(e)
        return
    for key, prompt in prompts.items():
        # Safely tweak instruction
        if hasattr(prompt, "instruction") and instruction_suffix:
            prompt.instruction = (prompt.instruction or "") + "\n" + instruction_suffix
        # Optionally patch examples (few-shots)
        if example_patch is not None and hasattr(prompt, "examples"):
            prompt.examples = example_patch
    metric.set_prompts(**prompts)  # write back


def dump_metric_prompts(metric, out_dir: Path):
    """
    Save *all* prompts used by a metric into a single file named <metric_name>.txt.
    If a metric has multiple prompts (e.g., single_turn_prompt, nli_prompt), we concatenate them
    with headers so you have the full, exact text the judge sees.
    """
    try:
        prompts: Dict[str, Any] = metric.get_prompts()
    except Exception as e:
        print(e)
        return
    parts = [f"# metric: {metric.name}"]
    for key, prompt in prompts.items():
        # Prefer to_string() if available; otherwise serialize fields
        if hasattr(prompt, "to_string"):
            text = prompt.to_string()
        else:
            # Fallback: dump instruction + examples if present
            instr = getattr(prompt, "instruction", "")
            ex = getattr(prompt, "examples", None)
            text = f"INSTRUCTION:\n{instr}\n\nEXAMPLES:\n{json.dumps(ex, indent=2, ensure_ascii=False)}"
        parts.append(f"\n## prompt key: {key}\n{text}\n")
    out_path = out_dir / f"{metric.name}.txt"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("ragas_prompts_out"), help="Directory to write <metric>.txt prompt files")
    parser.add_argument("--model", type=str, default=os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini"),
                        help="Judge LLM model name (OpenAI-compatible). Defaults to gpt-4o-mini")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", None),
                        help="OpenAI-compatible base URL (set for vLLM, Ollama+server, etc.)")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key (or compatible). Required unless your stack ignores it.")
    parser.add_argument("--device", type=str, default=os.getenv("FAITH_DEVICE", "cpu"),
                        help="Device for FaithfulnesswithHHEM (cpu/cuda).")
    args = parser.parse_args()

    # Ensure output dir
    args.out.mkdir(parents=True, exist_ok=True)

    # Configure the judge LLM for metrics (Ragas will inject embeddings automatically if not given)
    # llm_factory supports base_url for OpenAI-compatible gateways (e.g., vLLM).

    # TODO
    # evaluator_llm = llm_factory(model=args.model, base_url=args.base_url)
    evaluator_llm = None

    # Build dataset (replace with your real traces)
    dataset = build_example_dataset()

    # ---- Instantiate metrics with varied parameters ----
    # 1) Answer/Response Relevancy: stricter than default (default strictness=3)
    m_answer_rel = ResponseRelevancy(strictness=5, llm=evaluator_llm, name="answer_relevancy")

    # 2) Faithfulness (classic)
    m_faith = Faithfulness(llm=evaluator_llm, name="faithfulness")

    # 3) Faithfulness with HHEM (batched NLI, device configurable)
    m_faith_hhem = FaithfulnesswithHHEM(llm=evaluator_llm, name="faithfulness_with_hhem", device=args.device, batch_size=16)

    # 4) LLM Context Precision (uses reference answer to judge retrieved_contexts)
    m_ctx_prec = LLMContextPrecisionWithReference(llm=evaluator_llm, name="llm_context_precision_with_reference")

    # 5) LLM Context Recall (claims from reference checked against retrieved_contexts)
    m_ctx_recall = LLMContextRecall(llm=evaluator_llm, name="context_recall")

    # 6) Factual Correctness (claims + NLI), bias toward precision, high atomicity & coverage
    m_fact = FactualCorrectness(
        llm=evaluator_llm,
        name="factual_correctness",
        mode="precision",      # "precision" | "recall" | "f1"
        beta=0.5,              # <1 favors precision
        atomicity="high",      # "high" | "low"
        coverage="high"        # "high" | "low"
    )

    # 7) Semantic Similarity (embedding-only; not an LLM judge but useful as a sanity metric)
    m_sem = SemanticSimilarity(name="semantic_similarity", threshold=0.80)

    # 8) Aspect Critic – example: forbid hallucinated citations (binary critic with majority vote)
    m_aspect = AspectCritic(
        name="no_fabricated_citations",
        definition="Does the response invent sources, quotes, or citations not grounded in the provided contexts or reference?",
        llm=evaluator_llm,
        strictness=3,  # odd counts -> majority vote
    )

    metrics = [
        m_answer_rel,
        m_faith,
        m_faith_hhem,
        m_ctx_prec,
        m_ctx_recall,
        m_fact,
        m_sem,
        m_aspect,
    ]


    for metric in metrics:
        dump_metric_prompts(metric, Path("../prompts"))
    # ---- Customize prompts per metric (domain bias, output formatting, examples) ----
    # Keep these succinct; overfitting the judge prompt can distort scores.
    domain_suffix = "Domain: general QA. Be strict. Output minimal JSON when applicable."
    customize_metric_prompts(m_answer_rel, instruction_suffix=domain_suffix)
    customize_metric_prompts(m_faith, instruction_suffix="If context does not support a claim, mark it unfaithful.")
    customize_metric_prompts(m_faith_hhem, instruction_suffix="Prefer short rationales; focus on explicit support in context.")
    customize_metric_prompts(m_ctx_prec, instruction_suffix="Consider a chunk relevant only if it directly helps answer the user_input.")
    customize_metric_prompts(m_ctx_recall, instruction_suffix="Decompose the reference into atomic claims before judging recall.")
    customize_metric_prompts(m_fact, instruction_suffix="Decompose both response and reference into atomic claims before NLI.")
    customize_metric_prompts(m_sem, instruction_suffix="Use semantic overlap as a soft signal; avoid penalizing rephrasing.")
    customize_metric_prompts(m_aspect, instruction_suffix="Return 1 if any fabricated citation exists, else 0.")

    # ---- Persist the effective prompts (one file per metric) ----
    written = []
    for m in metrics:
        p = dump_metric_prompts(m, args.out)
        written.append(str(p))
    print("[prompts written]")
    for p in written:
        print(" -", p)

    # ---- Run evaluation ----
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,   # global default; per-metric llm already set
        # embeddings=None -> Ragas uses default OpenAI embeddings internally
        show_progress=True,
    )

    # Aggregate scores (dict-like)
    print("\n[aggregate scores]")
    print(json.dumps(result, indent=2))

    # Per-sample breakdown (to pandas)
    df = result.to_pandas()  # columns: metric scores per row + maybe cost if configured
    print("\n[per-sample scores]")
    with pd_option_context():
        # pretty print without dragging pandas into global imports
        pass


# small context manager to avoid global pandas import if you don’t want it.
from contextlib import contextmanager
@contextmanager
def pd_option_context():
    try:
        import pandas as pd  # type: ignore
        from pandas import option_context
        with option_context('display.max_columns', None, 'display.width', 160):
            yield
    except Exception:
        yield


if __name__ == "__main__":
    # Guard: require API key unless your base_url ignores it
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_BASE_URL"):
        print("Warning: OPENAI_API_KEY is not set. If your base_url requires it, set it before running.")

    main()
