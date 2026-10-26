from itertools import pairwise

from pgcite.chunking import chunk_text, load_markdown, split_sentences

TEXT = " ".join(f"Sentence {i} carries exactly six words." for i in range(40))


def test_split_sentences_strips_markers():
    text = "# Limits\n- First item. Second one!\n> Quoted line?\n\n1. Numbered step.\n---\n"
    text += "**Bold** stays. Short names, e.g. ops, are fine."
    assert split_sentences(text) == [
        "Limits",
        "First item.",
        "Second one!",
        "Quoted line?",
        "Numbered step.",
        "**Bold** stays.",
        "Short names, e.g. ops, are fine.",
    ]


def test_chunks_respect_word_limit():
    chunks = chunk_text(TEXT, max_words=30, overlap=0)
    assert len(chunks) == 8
    assert all(len(chunk.split()) <= 30 for chunk in chunks)
    assert [s for chunk in chunks for s in split_sentences(chunk)] == split_sentences(TEXT)


def test_headings_stay_separate_sentences_inside_chunks():
    chunks = chunk_text("# Limits by plan\n- Team gets 300 requests per minute.\n- Free gets 60")
    assert split_sentences(chunks[0]) == [
        "Limits by plan",
        "Team gets 300 requests per minute.",
        "Free gets 60",
    ]


def test_overlap_repeats_last_sentence():
    chunks = chunk_text(TEXT, max_words=30, overlap=1)
    assert all(len(chunk.split()) <= 30 for chunk in chunks)
    for previous, current in pairwise(chunks):
        assert current.startswith(split_sentences(previous)[-1])


def test_long_sentence_becomes_its_own_chunk():
    long_sentence = " ".join(["word"] * 50) + "."
    chunks = chunk_text(f"Short one here. {long_sentence} Another short one.", max_words=20)
    assert chunks == ["Short one here.", long_sentence, "Another short one."]


def test_no_empty_chunks():
    assert chunk_text("") == []
    assert chunk_text("\n  \n- \n") == []
    assert all(chunk.strip() for chunk in chunk_text(TEXT, max_words=5))


def test_load_markdown(tmp_path):
    (tmp_path / "billing.md").write_text("# Billing basics\n\nInvoices go out monthly.\n")
    (tmp_path / "about.md").write_text("No heading in this one.\n")
    (tmp_path / "notes.txt").write_text("Not markdown.")
    documents = load_markdown(tmp_path)
    assert [doc.id for doc in documents] == ["about", "billing"]
    assert documents[0].title == "about"
    assert documents[1].title == "Billing basics"
    assert documents[1].text == "Invoices go out monthly."
