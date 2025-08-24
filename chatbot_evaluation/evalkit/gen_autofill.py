# chatbot_evaluation/evalkit/gen_autofill.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Iterable
import json, time
import requests

from chatbot_evaluation.evalkit.files import write_jsonl

class ChatClientHTTP:
    def __init__(self, base_url: str, timeout_s: int = 30, headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_s
        self.headers = headers or {"Content-Type": "application/json"}

    def answer(self, query: str, contexts: List[str] | None = None) -> str:
        """
        Adapt to your API contract.
        Expected to return the model's final answer text.
        """
        payload = {"query": query, "contexts": contexts or []}
        r = requests.post(f"{self.base_url}/chat", json=payload, timeout=self.timeout, headers=self.headers)
        r.raise_for_status()
        data = r.json()
        return data.get("answer") or data.get("response") or ""

def autofill_answers_for_dataset(
    in_jsonl: str | Path,
    out_jsonl: str | Path,
    chat_client: ChatClientHTTP,
    contexts_map: Dict[str, List[str]] | None = None,
) -> None:
    rows_out = []
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            qid = str(row.get("qid") or row.get("id") or "")
            query = row.get("query") or row.get("user_input") or ""
            response = row.get("response") or ""
            ctx = row.get("retrieved_contexts") or (contexts_map.get(qid) if contexts_map else None)

            if not response:
                try:
                    response = chat_client.answer(query=query, contexts=ctx or [])
                except Exception as e:
                    response = ""  # keep going
                    row.setdefault("_errors", {})["chat"] = str(e)

            row["response"] = response
            if ctx:
                row["retrieved_contexts"] = ctx
            rows_out.append(row)

    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_jsonl, rows_out)