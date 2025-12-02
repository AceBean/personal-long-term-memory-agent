from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "about",
    "as",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "this",
    "that",
    "it",
    "its",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
    counts = Counter(tokenize(text))
    return [word for word, _ in counts.most_common(max_keywords)]


def similarity(query: Sequence[str], tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    query_set = set(query)
    token_set = set(tokens)
    intersection = len(query_set & token_set)
    return intersection / max(len(query_set), 1)


def generate_summary(contents: Iterable[str], max_sentences: int = 3) -> str:
    sentences: List[str] = []
    for text in contents:
        sentences.extend(re.split(r"(?<=[.!?]) +", text.strip()))
    sentences = [s for s in sentences if s]
    return " ".join(sentences[:max_sentences])


def answer_question(question: str, context: Iterable[str]) -> str:
    question_tokens = tokenize(question)
    if not question_tokens:
        return "I need a clearer question to answer."

    best_sentence = None
    best_score = 0.0
    for text in context:
        for sentence in re.split(r"(?<=[.!?]) +", text.strip()):
            score = similarity(question_tokens, tokenize(sentence))
            if score > best_score:
                best_score = score
                best_sentence = sentence

    if best_sentence:
        return best_sentence
    return "I could not find a confident answer, but here is what I know: " + generate_summary(context, max_sentences=2)
