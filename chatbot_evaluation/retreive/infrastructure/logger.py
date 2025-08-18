import logging
import sys
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def log_manifest(run_config: dict, results: dict, path: str):
    import json, os
    os.makedirs("runs", exist_ok=True)
    filename = f"runs/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(filename, "w") as f:
        json.dump({"config": run_config, "results": results}, f, indent=2)
    return filename
