# infrastructure/pytrec_evaluator.py
import pytrec_eval
from retreive.domain.interfaces import Evaluator

class PytrecEvaluator(Evaluator):
    def evaluate(self, run, qrels, metrics):
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
        return evaluator.evaluate(run)
