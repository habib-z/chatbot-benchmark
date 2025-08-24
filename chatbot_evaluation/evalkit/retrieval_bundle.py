# chatbot_evaluation/evalkit/retrieval_bundle.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import unicodedata, re, yaml

# ---------- dataclasses ----------
@dataclass
class EncoderCfg:
    model_id: str
    normalize_embeddings: bool
    max_length: int
    query_prefix: str
    query_suffix: str
    passage_prefix: str
    passage_suffix: str

@dataclass
class QdrantCfg:
    host: str
    port: int
    prefer_grpc: bool
    collection_prefix: str
    metric: str  # "cosine" | "dot" | "l2" | etc.

@dataclass
class CorpusCfg:
    id: str
    path: str
    id_field: str
    text_field: str

@dataclass
class RetrievalBundle:
    encoder: EncoderCfg
    qdrant: QdrantCfg
    corpus: CorpusCfg
    defaults_top_k: int
    nfkc: bool
    strip: bool
    collapse_ws: bool
    norm_yeh: bool
    norm_kaf: bool
    remove_tatweel: bool
    zwnj_mode: str

# ---------- loader ----------
def load_bundle(path: str | Path) -> RetrievalBundle:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))

    enc = data["encoder"]
    tn  = data["text_normalization"]
    pp  = enc.get("prompt_profiles", {}) or {}
    index = data.get("index", {}) or {}
    qd = index.get("qdrant", {}) or {}
    corp = data["corpus"]
    defs = data.get("defaults", {}) or {}

    return RetrievalBundle(
        encoder=EncoderCfg(
            model_id=enc["model_id"],
            normalize_embeddings=bool(enc.get("normalize_embeddings", True)),
            max_length=int(enc.get("max_length", 512)),
            query_prefix=(pp.get("query", {}) or {}).get("prefix", ""),
            query_suffix=(pp.get("query", {}) or {}).get("suffix", ""),
            passage_prefix=(pp.get("passage", {}) or {}).get("prefix", ""),
            passage_suffix=(pp.get("passage", {}) or {}).get("suffix", ""),
        ),
        qdrant=QdrantCfg(
            host=str(qd["host"]),
            port=int(qd["port"]),
            prefer_grpc=bool(qd.get("prefer_grpc", True)),
            collection_prefix=str(qd.get("collection_prefix", "retrieval")),
            metric=str(index.get("metric", "cosine")).lower(),
        ),
        corpus=CorpusCfg(
            id=str(corp["id"]),
            path=str(corp["path"]),
            id_field=str(corp["doc_schema"]["id_field"]),
            text_field=str(corp["doc_schema"]["text_field"]),
        ),
        defaults_top_k=int(defs.get("retrieval_top_k", 5)),
        nfkc=bool(tn.get("unicode_nfkc", True)),
        strip=bool(tn.get("strip", True)),
        collapse_ws=bool(tn.get("collapse_whitespace", True)),
        norm_yeh=bool((tn.get("persian_char_normalization", {}) or {}).get("normalize_arabic_yeh_to_persian_ye", True)),
        norm_kaf=bool((tn.get("persian_char_normalization", {}) or {}).get("normalize_arabic_kaf_to_persian_ke", True)),
        remove_tatweel=bool((tn.get("persian_char_normalization", {}) or {}).get("remove_tatweel", True)),
        zwnj_mode=str((tn.get("persian_char_normalization", {}) or {}).get("normalize_zwnj", "keep")),
    )

# ---------- text normalizer ----------
_AR_YEH="\u064A"; _FA_YE="\u06CC"; _AR_KAF="\u0643"; _FA_KE="\u06A9"; _TATWEEL="\u0640"; _ZWNJ="\u200C"

def make_normalizer(rb: RetrievalBundle) -> Callable[[str], str]:
    def norm(s: str) -> str:
        if s is None:
            return ""
        x = s
        if rb.nfkc: x = unicodedata.normalize("NFKC", x)
        if rb.norm_yeh: x = x.replace(_AR_YEH, _FA_YE)
        if rb.norm_kaf: x = x.replace(_AR_KAF, _FA_KE)
        if rb.remove_tatweel: x = x.replace(_TATWEEL, "")
        if rb.zwnj_mode == "remove":
            x = x.replace(_ZWNJ, "")
        elif rb.zwnj_mode == "normalize":
            x = re.sub(r"[ \t]+", _ZWNJ, x)
        if rb.strip: x = x.strip()
        if rb.collapse_ws: x = re.sub(r"\s+", " ", x)
        return x
    return norm

# ---------- factories for infra ----------
def build_embedder(rb: RetrievalBundle, device: str | None = None):
    """
    Returns your SentenceEmbedder configured from the bundle.
    NOTE: We keep prefixes/suffixes OUT of the embedder here, to avoid touching ingest/retrieve code.
          If you later want E5-style prompts, add them where you prepare the text.
    """
    from chatbot_evaluation.retreive.infrastructure.sentence_embedder import SentenceEmbedder
    return SentenceEmbedder(
        rb.encoder.model_id,
        device=device,
        # if your class supports: normalize=rb.encoder.normalize_embeddings, max_length=rb.encoder.max_length
    )

def build_qdrant_repo(rb: RetrievalBundle):
    from chatbot_evaluation.retreive.infrastructure.qdrant_repo import QdrantRepository
    try:
        return QdrantRepository(
            host=rb.qdrant.host,
            port=rb.qdrant.port,
            prefer_grpc=getattr(rb.qdrant, "prefer_grpc", True),
        )
    except TypeError:
        return QdrantRepository(host=rb.qdrant.host, port=rb.qdrant.port)

# ---------- scoring helper ----------
def score_mode_from_bundle(rb: RetrievalBundle) -> str:
    """
    Returns 'similarity' if metric is cosine/dot/ip, else 'distance' for l2/euclidean.
    Used so pytrec side always sees 'higher is better'.
    """
    m = (rb.qdrant.metric or "").lower()
    if m in ("cosine", "dot", "inner_product", "ip"):
        return "similarity"
    if m in ("l2", "euclidean"):
        return "distance"
    # default safe choice
    return "similarity"