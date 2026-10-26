import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from pgcite.config import Settings, get_settings


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


class Embedder:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(next(iter(self.model.query_embed(text))))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack(list(self.model.passage_embed(texts))))


class Reranker:
    def __init__(self, model_name: str, cache_dir: str | None = None):
        self.model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [float(score) for score in self.model.rerank(query, passages)]


def load_models(settings: Settings) -> tuple[Embedder, Reranker]:
    embedder = Embedder(settings.embedding_model, settings.model_cache_dir)
    reranker = Reranker(settings.reranker_model, settings.model_cache_dir)
    return embedder, reranker


if __name__ == "__main__":
    settings = get_settings()
    embedder, reranker = load_models(settings)
    dim = embedder.embed_query("warm-up").shape[0]
    reranker.score("warm-up", ["warm-up"])
    print(f"{settings.embedding_model} ({dim} dims) and {settings.reranker_model} ready")
