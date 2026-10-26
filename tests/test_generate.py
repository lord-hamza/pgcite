from types import SimpleNamespace

import pytest

from pgcite.generate import (
    NO_ANSWER,
    REFUSAL,
    AnthropicGenerator,
    ExtractiveGenerator,
    cost_usd,
    extract_citations,
    normalize_citations,
    strip_citations,
)
from pgcite.search import Hit

HITS = [
    Hit(11, "pricing", "Pricing", "The Team plan costs 12 dollars per seat per month.", 0.9),
    Hit(12, "storage", "Storage", "Each Team workspace gets 500 GB of shared storage.", 0.8),
    Hit(13, "sso", "SSO", "SAML single sign-on is part of the Enterprise plan.", 0.7),
]


def test_normalize_citations():
    assert normalize_citations("Foo. [1] Bar.") == "Foo [1]. Bar."
    assert normalize_citations("Foo.[1][2] Bar") == "Foo [1][2]. Bar"
    assert normalize_citations("Already fine [1].") == "Already fine [1]."


def test_strip_citations():
    assert (
        strip_citations("Twelve dollars [1][2]. Two months [3].") == "Twelve dollars. Two months."
    )


def test_extract_citations_dedupes_and_ignores_out_of_range():
    citations = extract_citations("A [2]. B [1][2]. C [7]. D [0].", HITS)
    assert [(c.ref, c.doc_id) for c in citations] == [(2, "storage"), (1, "pricing")]
    assert citations[0].chunk_id == 12


def test_extractive_generator_cites_closest_sentence(embedder):
    generator = ExtractiveGenerator(embedder, max_sentences=1)
    answer = generator.generate("How much does the Team plan cost per seat?", HITS)
    assert answer.text == "The Team plan costs 12 dollars per seat per month [1]."
    assert [c.doc_id for c in answer.citations] == ["pricing"]
    assert (answer.model, answer.cost_usd) == ("extractive", 0.0)


def test_extractive_generator_keeps_rank_order(embedder):
    answer = ExtractiveGenerator(embedder).generate("Team plan storage and price per seat", HITS)
    assert answer.text.index("[1]") < answer.text.index("[2]")


def test_extractive_generator_skips_headings_and_repeats(embedder):
    overlap = "Each Team workspace gets 500 GB of shared storage."
    hits = [
        Hit(21, "storage", "Storage", f"Storage limits per Team workspace\n{overlap}", 0.9),
        Hit(22, "storage", "Storage", f"{overlap}\nFiles up to 2 GB can be uploaded.", 0.8),
    ]
    answer = ExtractiveGenerator(embedder).generate("How much storage does Team get?", hits)
    assert answer.text.count("500 GB") == 1
    assert "Storage limits" not in answer.text
    assert answer.text.startswith("Each Team workspace gets 500 GB of shared storage [1].")


def test_extractive_generator_without_hits(embedder):
    answer = ExtractiveGenerator(embedder).generate("Anything?", [])
    assert (answer.text, answer.citations) == (NO_ANSWER, [])


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def fake_generator(model="claude-opus-5-5", served_by=None, stop_reason="end_turn", texts=()):
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=text) for text in texts]
    messages = FakeMessages(
        SimpleNamespace(
            model=served_by or model,
            stop_reason=stop_reason,
            content=content,
            usage=SimpleNamespace(input_tokens=1200, output_tokens=300),
        )
    )
    client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicGenerator(model, 2048, "low", client=client), messages.calls


def test_anthropic_generator_reads_text_blocks_and_prices_usage():
    texts = ["The Team plan costs 12 dollars per seat [1].", " It has 500 GB [2]."]
    generator, calls = fake_generator(texts=texts)
    answer = generator.generate("How much is Team?", HITS)
    assert answer.text == "The Team plan costs 12 dollars per seat [1]. It has 500 GB [2]."
    assert [c.doc_id for c in answer.citations] == ["pricing", "storage"]
    assert (answer.input_tokens, answer.output_tokens) == (1200, 300)
    assert answer.cost_usd == pytest.approx((1200 * 4.00 + 300 * 20.00) / 1_000_000)

    request = calls[0]
    assert request["messages"][0]["content"].startswith("[1] Pricing\nThe Team plan costs")
    assert request["output_config"] == {"effort": "low"}
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["fallbacks"] == "default"
    assert not {"temperature", "top_p", "top_k", "thinking"} & request.keys()


def test_fallback_params_only_for_supported_models():
    generator, calls = fake_generator(model="claude-haiku-5-5", texts=["Hi [1]."])
    answer = generator.generate("q", HITS)
    assert "fallbacks" not in calls[0] and "betas" not in calls[0]
    assert answer.cost_usd == pytest.approx(cost_usd("claude-haiku-5-5", 1200, 300))
    assert cost_usd("some-local-model", 1200, 300) == 0.0


def test_unknown_serving_model_is_priced_as_configured_model():
    generator, _ = fake_generator(served_by="claude-opus-4-8", texts=["Hi [1]."])
    answer = generator.generate("q", HITS)
    assert answer.model == "claude-opus-4-8"
    assert answer.cost_usd == pytest.approx(cost_usd("claude-opus-5-5", 1200, 300))


def test_refusal_returns_fixed_message():
    generator, _ = fake_generator(stop_reason="refusal")
    answer = generator.generate("q", HITS)
    assert (answer.text, answer.citations) == (REFUSAL, [])
