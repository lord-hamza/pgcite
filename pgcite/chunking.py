import re
from dataclasses import dataclass
from pathlib import Path

SENTENCE_BREAK = re.compile(r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<=[.!?])\s+")
LINE_MARKER = re.compile(r"^(?:\s*(?:[-*>]|#+|\d+[.)])\s+)+")
WORD = re.compile(r"\w")


@dataclass
class Document:
    id: str
    title: str
    text: str


def load_markdown(directory: Path) -> list[Document]:
    documents = []
    for path in sorted(Path(directory).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = path.stem
        for index, line in enumerate(lines):
            if line.startswith("# "):
                title = line[2:].strip()
                del lines[index]
                break
        documents.append(Document(id=path.stem, title=title, text="\n".join(lines).strip()))
    return documents


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in text.splitlines():
        line = LINE_MARKER.sub("", line).strip()
        if WORD.search(line):
            sentences.extend(SENTENCE_BREAK.split(line))
    return sentences


def chunk_text(text: str, max_words: int = 120, overlap: int = 1) -> list[str]:
    # One sentence per line, so split_sentences(chunk) gives back the original sentences and
    # headings or unpunctuated list items never merge into the sentence after them.
    chunks: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text):
        size = len(sentence.split())
        if current and _word_count(current) + size > max_words:
            chunks.append("\n".join(current))
            current = current[-overlap:] if overlap > 0 else []
            if _word_count(current) + size > max_words:
                current = []
        current.append(sentence)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _word_count(sentences: list[str]) -> int:
    return sum(len(sentence.split()) for sentence in sentences)
