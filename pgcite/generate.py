import re
from dataclasses import dataclass
from typing import Protocol

import anthropic
import numpy as np

from pgcite.chunking import split_sentences
from pgcite.config import Settings
from pgcite.models import Embedder
from pgcite.search import Hit

PRICES_PER_MTOK = {  # USD per million tokens (input, output), Anthropic first-party pricing
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
NO_ANSWER = "I couldn't find that in the indexed documents."
REFUSAL = "The model declined to answer this question."
CITATION = re.compile(r"\[(\d+)\]")
TRAILING_CITATIONS = re.compile(r"\s*([.!?])\s*((?:\[\d+\]\s*)*\[\d+\])")

SYSTEM_PROMPT = (
    "Answer the question in two to four sentences, using only the numbered passages. Cite the "
    "passages that support each sentence as [1] or [2][3], placed before the sentence's final "
    f"punctuation. If the passages do not contain the answer, reply exactly: {NO_ANSWER}"
)


@dataclass
class Citation:
    ref: int
    chunk_id: int
    doc_id: str
    title: str
    snippet: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Generator(Protocol):
    name: str

    def generate(self, question: str, hits: list[Hit]) -> Answer: ...


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = PRICES_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def normalize_citations(text: str) -> str:
    return TRAILING_CITATIONS.sub(r" \2\1", text)


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def extract_citations(text: str, hits: list[Hit]) -> list[Citation]:
    citations: list[Citation] = []
    for ref in dict.fromkeys(int(number) for number in CITATION.findall(text)):
        if 1 <= ref <= len(hits):
            hit = hits[ref - 1]
            citations.append(
                Citation(ref, hit.chunk_id, hit.doc_id, hit.title, _snippet(hit.content))
            )
    return citations


def _snippet(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def build_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n\n".join(f"[{i}] {hit.title}\n{hit.content}" for i, hit in enumerate(hits, 1))
    return f"{passages}\n\nQuestion: {question}"


class ExtractiveGenerator:
    """Answers with the passage sentences closest to the question. Offline and free.

    Only complete sentences (five or more words, ending in . ! or ?) are candidates, which
    skips headings and bare lists, and a sentence repeated by chunk overlap is used once.
    """

    name = "extractive"

    def __init__(self, embedder: Embedder, max_sentences: int = 2):
        self.embedder = embedder
        self.max_sentences = max_sentences

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        refs: dict[str, int] = {}
        for ref, hit in enumerate(hits[:3], 1):
            for sentence in split_sentences(hit.content):
                if len(sentence.split()) >= 5 and sentence[-1] in ".!?":
                    refs.setdefault(sentence, ref)
        if not refs:
            return Answer(NO_ANSWER, [], self.name)
        sentences = list(refs)
        similarity = self.embedder.embed_passages(sentences) @ self.embedder.embed_query(question)
        best = sorted(np.argsort(-similarity, kind="stable")[: self.max_sentences])
        chosen = [sentences[i] for i in best]
        text = " ".join(f"{sentence[:-1]} [{refs[sentence]}]{sentence[-1]}" for sentence in chosen)
        return Answer(text, extract_citations(text, hits), self.name)


class AnthropicGenerator:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        max_tokens: int,
        effort: str,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(NO_ANSWER, [], self.model)
        fallback = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
            output_config={"effort": self.effort},
            **(fallback if self.model in FALLBACK_MODELS else {}),
        )
        if response.stop_reason == "refusal":
            text = REFUSAL
        else:
            text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        priced_as = response.model if response.model in PRICES_PER_MTOK else self.model
        return Answer(
            text=text.strip(),
            citations=extract_citations(text, hits),
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost_usd(priced_as, usage.input_tokens, usage.output_tokens),
        )


def build_generator(settings: Settings, embedder: Embedder) -> Generator:
    if settings.generator == "anthropic":
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        return AnthropicGenerator(
            settings.llm_model, settings.llm_max_tokens, settings.llm_effort, client
        )
    return ExtractiveGenerator(embedder)
