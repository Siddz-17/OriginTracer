"""Optional local models (CPU, no API key): sentence embeddings for claim matching and NLI for stance.

    pip install sentence-transformers    (+ torch; CPU wheel: pip install torch --index-url https://download.pytorch.org/whl/cpu)

TRACER_LOCAL_MODELS=auto (default) uses them when the package is installed, 0 disables them, 1 requires them.
Models download once from Hugging Face (~25 MB embedder, ~140 MB NLI) and are cached.

How they are used (thresholds come from probing real headlines, see README "Local models"):
  * embeddings  keyword matches whose meaning is far from the claim (similarity < MATCH_MIN_SIM) are dropped
                ("Bill Gates did not promote water cryptocoin" shares the name, not the claim); paraphrases
                with few shared keywords but high similarity (>= RESCUE_SIM) are added.
  * NLI         only ADDS debunks: contradiction >= NLI_CONTRA with similarity >= NLI_MIN_SIM. On its own the
                small NLI model also calls unrelated text "contradiction", so it is never trusted without
                high similarity, and it never overrides the rating / cue-based stance rules.
"""
import logging
import os

logger = logging.getLogger(__name__)

EMBED_MODEL = os.getenv("TRACER_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
NLI_MODEL = os.getenv("TRACER_NLI_MODEL", "cross-encoder/nli-deberta-v3-xsmall")
MATCH_MIN_SIM = float(os.getenv("TRACER_MATCH_MIN_SIM", "0.45"))
RESCUE_SIM = float(os.getenv("TRACER_RESCUE_SIM", "0.65"))
NLI_MIN_SIM = float(os.getenv("TRACER_NLI_MIN_SIM", "0.7"))
NLI_CONTRA = float(os.getenv("TRACER_NLI_CONTRA", "0.8"))

_models = {}
_failed = {"v": False}


def enabled():
    mode = os.getenv("TRACER_LOCAL_MODELS", "auto").lower()
    if mode in ("0", "false", "off") or _failed["v"]:
        return False
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        if mode in ("1", "true", "on"):
            raise RuntimeError("TRACER_LOCAL_MODELS=1 but sentence-transformers is not installed")
        return False


def _load(kind):
    if kind not in _models:
        from sentence_transformers import CrossEncoder, SentenceTransformer
        try:
            _models[kind] = (SentenceTransformer(EMBED_MODEL, device="cpu") if kind == "embed"
                             else CrossEncoder(NLI_MODEL, device="cpu"))
        except Exception as e:  # offline / download failure: fall back to the keyword heuristics
            logger.warning("local model %s unavailable (%s); using heuristics only", kind, e)
            _failed["v"] = True
            raise
    return _models[kind]


def similarities(claim, texts):
    """Cosine similarity of each text (headline / lead) to the claim, or None if models are off."""
    if not texts or not enabled():
        return None
    try:
        m = _load("embed")
    except Exception:
        return None
    vecs = m.encode([claim] + [t[:300] for t in texts], normalize_embeddings=True, batch_size=64,
                    show_progress_bar=False)
    return [float(v) for v in vecs[1:] @ vecs[0]]


def contradiction(claim, texts):
    """P(text contradicts claim) per text from the NLI cross-encoder, or None if models are off."""
    if not texts or not enabled():
        return None
    try:
        m = _load("nli")
    except Exception:
        return None
    probs = m.predict([(t[:300], claim) for t in texts], apply_softmax=True, batch_size=32,
                      show_progress_bar=False)
    idx = {v.lower(): k for k, v in m.config.id2label.items()}["contradiction"]
    return [float(p[idx]) for p in probs]
