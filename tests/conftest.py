import re
import zlib

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

from pgcite.api import app
from pgcite.config import get_settings

TOKEN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag of words: deterministic, instant, and good enough to rank overlapping text."""

    dim = 256

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in TOKEN.findall(text.lower()):
            vector[zlib.crc32(token.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.array([self._embed(text) for text in texts]).reshape(len(texts), self.dim)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(scope="session")
def client():
    url = get_settings().database_url
    try:
        psycopg.connect(url, connect_timeout=2).close()
    except psycopg.OperationalError:
        pytest.skip(f"Postgres is not reachable at {url}")
    with TestClient(app) as test_client:
        yield test_client
