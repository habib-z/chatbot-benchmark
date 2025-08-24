from typing import Dict, Any, Optional

class Reporter:
    def start_run(self, manifest: Dict[str,Any]): ...
    def log_artifact(self, path: str, name: Optional[str]=None): ...
    def log_metrics(self, metrics: Dict[str,float]): ...
    def end_run(self): ...

class NoopReporter(Reporter):
    def start_run(self, manifest): pass
    def log_artifact(self, path, name=None): pass
    def log_metrics(self, metrics): pass
    def end_run(self): pass

class MLflowReporter(Reporter):
    def __init__(self, experiment: str = "rag-eval"):
        import mlflow
        self.mlflow = mlflow
        self.experiment = experiment
        self.active = None

    def start_run(self, manifest):
        self.mlflow.set_experiment(self.experiment)
        self.active = self.mlflow.start_run(run_name=manifest["run"]["id"])
        self.mlflow.log_params({
            "model": manifest["model_under_test"]["name"],
            "dataset": manifest["dataset"]["name"],
        })

    def log_artifact(self, path, name=None):
        self.mlflow.log_artifact(path, artifact_path=name)

    def log_metrics(self, metrics):
        self.mlflow.log_metrics(metrics)

    def end_run(self):
        if self.active:
            self.mlflow.end_run()
            self.active = None