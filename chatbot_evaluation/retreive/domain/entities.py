# domain/entities.py
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Document:
    doc_id: str
    text: str

@dataclass
class Query:
    query_id: str
    text: str

# Relevance judgements: {query_id: {doc_id: score}}
Qrels = Dict[str, Dict[str, int]]
