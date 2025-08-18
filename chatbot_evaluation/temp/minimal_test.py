
import pandas as pd
from ragas.dataset_schema import SingleTurnSample
from chatbot_evaluation.temp.judge import get_local_judge
from ragas.metrics import ResponseGroundedness, Faithfulness

evaluator_llm = get_local_judge()

df = pd.read_json("../benchmarks/asiyeh/gemma-12b-ref-context.jsonl", lines=True)

samples = [
    SingleTurnSample(
        user_input=row["query"],
        response=row["response"],
        reference = row["reference"],
        retrieved_contexts=[row["retrieved_contexts"]]
    )
    for _, row in df.iterrows()
]
samples=samples[:2]

# metric_ans_cor = AnswerCorrectness(answer_similarity=None, llm=evaluator_llm, weights=[1,0]),
metric_res_groud = ResponseGroundedness(llm=evaluator_llm)
metric_faith = Faithfulness(llm=evaluator_llm)

scores=[]
for sample in samples:
    score=metric_faith.single_turn_score(sample)
    scores.append(score)
    print(score)

print(scores)