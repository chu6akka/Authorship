from __future__ import annotations

import argparse
import dataclasses
import re
from collections import Counter
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_]+", re.UNICODE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
EMOTICON_RE = re.compile(r"(:\)|:\(|;\)|:D|XD|xD|<3)")

SYNONYM_GROUPS = [
    {"бежать", "бегать", "нестись"},
    {"классно", "круто", "огонь"},
    {"говорить", "сказать", "писать"},
    {"сейчас", "щас", "теперь"},
    {"ненавидеть", "не любить"},
]

LAT_TO_CYR = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
    "j": "й", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс", "y": "ы", "z": "з",
}


@dataclasses.dataclass
class MatchResult:
    exact_sentences: list[str]
    ngram_matches: list[str]
    synonym_matches: list[str]
    paraphrase_pairs: list[tuple[str, str, float]]
    translit_matches: list[tuple[str, str]]
    internet_markers: dict[str, int]


@dataclasses.dataclass
class TextStats:
    words: int
    unique_words: int
    sentences: int
    paragraphs: int


@dataclasses.dataclass
class AnalysisResult:
    query_stats: TextStats
    sample_stats: TextStats
    match_result: MatchResult


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [s.strip() for s in SENT_SPLIT_RE.split(normalized) if s.strip()]


def tokenize_words(text: str) -> list[str]:
    return [t.lower() for t in WORD_RE.findall(text)]


def simple_ru_stem(token: str) -> str:
    suffixes = (
        "иями", "ями", "ами", "ого", "ему", "ому", "ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой",
        "ам", "ям", "ах", "ях", "ом", "ем", "ой", "ую", "юю", "а", "я", "ы", "и", "у", "ю", "е", "о",
    )
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def make_ngrams(tokens: list[str], n: int = 3) -> list[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def build_synonym_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        for word in group:
            index[word] = group
    return index


def translit_to_cyr(token: str) -> str:
    out = []
    for ch in token.lower():
        out.append(LAT_TO_CYR.get(ch, ch))
    return "".join(out)


def text_stats(text: str) -> TextStats:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = len([p for p in text.split("\n") if p.strip()])
    return TextStats(
        words=len(words),
        unique_words=len(set(words)),
        sentences=len(sentences),
        paragraphs=paragraphs,
    )


def find_exact_sentence_matches(query: str, sample: str) -> list[str]:
    q = set(split_sentences(query))
    s = set(split_sentences(sample))
    return sorted(q.intersection(s))


def find_ngram_matches(query: str, sample: str, n: int = 3, top_k: int = 20) -> list[str]:
    q_ngrams = Counter(make_ngrams([simple_ru_stem(t) for t in tokenize_words(query)], n=n))
    s_ngrams = Counter(make_ngrams([simple_ru_stem(t) for t in tokenize_words(sample)], n=n))
    shared = set(q_ngrams).intersection(s_ngrams)
    ranked = sorted(shared, key=lambda g: (q_ngrams[g] + s_ngrams[g], g), reverse=True)
    return ranked[:top_k]


def find_synonym_matches(query: str, sample: str) -> list[str]:
    syn_index = build_synonym_index()
    q_tokens = [simple_ru_stem(t) for t in tokenize_words(query)]
    s_tokens = [simple_ru_stem(t) for t in tokenize_words(sample)]
    matches = set()
    for qt in q_tokens:
        for syn_group in SYNONYM_GROUPS:
            if qt in syn_group:
                for st in s_tokens:
                    if st in syn_group and st != qt:
                        matches.add(f"{qt} ↔ {st}")
    return sorted(matches)


def sentence_similarity(a: str, b: str) -> float:
    a_set = {simple_ru_stem(t) for t in tokenize_words(a)}
    b_set = {simple_ru_stem(t) for t in tokenize_words(b)}
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set.intersection(b_set))
    union = len(a_set.union(b_set))
    return inter / union


def find_paraphrases(query: str, sample: str, threshold: float = 0.45, top_k: int = 10) -> list[tuple[str, str, float]]:
    q_sent = split_sentences(query)
    s_sent = split_sentences(sample)
    pairs: list[tuple[str, str, float]] = []
    for qs in q_sent:
        for ss in s_sent:
            score = sentence_similarity(qs, ss)
            if score >= threshold and qs != ss:
                pairs.append((qs, ss, round(score, 3)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def find_translit_matches(query: str, sample: str) -> list[tuple[str, str]]:
    q_tokens = tokenize_words(query)
    s_tokens = tokenize_words(sample)
    s_norm = {simple_ru_stem(t) for t in s_tokens}
    out = []
    for qt in q_tokens:
        if re.search(r"[a-zA-Z]", qt):
            cyr = simple_ru_stem(translit_to_cyr(qt))
            for st in s_norm:
                if cyr == st:
                    out.append((qt, st))
    return sorted(set(out))


def internet_markers(text: str) -> dict[str, int]:
    tokens = tokenize_words(text)
    return {
        "emoji_or_emoticons": len(EMOTICON_RE.findall(text)),
        "latin_tokens": sum(1 for t in tokens if re.search(r"[a-zA-Z]", t)),
        "elongated_words": sum(1 for t in tokens if re.search(r"(.)\1\1", t)),
        "excessive_punctuation": len(re.findall(r"[!?]{2,}", text)),
    }


def run_analysis(query_text: str, sample_text: str) -> AnalysisResult:
    return AnalysisResult(
        query_stats=text_stats(query_text),
        sample_stats=text_stats(sample_text),
        match_result=MatchResult(
            exact_sentences=find_exact_sentence_matches(query_text, sample_text),
            ngram_matches=find_ngram_matches(query_text, sample_text),
            synonym_matches=find_synonym_matches(query_text, sample_text),
            paraphrase_pairs=find_paraphrases(query_text, sample_text),
            translit_matches=find_translit_matches(query_text, sample_text),
            internet_markers={
                "query_emoji_or_emoticons": internet_markers(query_text)["emoji_or_emoticons"],
                "sample_emoji_or_emoticons": internet_markers(sample_text)["emoji_or_emoticons"],
                "query_latin_tokens": internet_markers(query_text)["latin_tokens"],
                "sample_latin_tokens": internet_markers(sample_text)["latin_tokens"],
            },
        ),
    )


def render_report(result: AnalysisResult, query_name: str, sample_name: str) -> str:
    m = result.match_result
    lines = [
        "# Аналитический отчет MVP по автороведческому сравнению",
        "",
        "## 1. Входные материалы",
        f"- Исследуемый текст: **{query_name}**",
        f"- Образец: **{sample_name}**",
        "",
        "## 2. Базовая статистика",
        f"- Исследуемый: слов={result.query_stats.words}, уникальных={result.query_stats.unique_words}, предложений={result.query_stats.sentences}, абзацев={result.query_stats.paragraphs}",
        f"- Образец: слов={result.sample_stats.words}, уникальных={result.sample_stats.unique_words}, предложений={result.sample_stats.sentences}, абзацев={result.sample_stats.paragraphs}",
        "",
        "## 3. Совпадения и частные признаки",
        f"- Полностью идентичные формулировки: **{len(m.exact_sentences)}**",
    ]
    lines.extend([f"  - {x}" for x in m.exact_sentences[:20]] or ["  - не обнаружены"])
    lines.append(f"- Совпадающие n-граммы: **{len(m.ngram_matches)}**")
    lines.extend([f"  - {x}" for x in m.ngram_matches[:20]] or ["  - не обнаружены"])
    lines.append(f"- Синонимические соответствия: **{len(m.synonym_matches)}**")
    lines.extend([f"  - {x}" for x in m.synonym_matches[:20]] or ["  - не обнаружены"])
    lines.append(f"- Перефраз (кандидаты): **{len(m.paraphrase_pairs)}**")
    lines.extend([f"  - [{score}] {a}  ⇄  {b}" for a, b, score in m.paraphrase_pairs[:20]] or ["  - не обнаружены"])
    lines.append(f"- Транслитерация (кандидаты): **{len(m.translit_matches)}**")
    lines.extend([f"  - {a} ↔ {b}" for a, b in m.translit_matches[:20]] or ["  - не обнаружены"])
    lines.append("")
    lines.append("## 4. Маркеры интернет-коммуникации")
    for k, v in m.internet_markers.items():
        lines.append(f"- {k}: {v}")

    lines.extend(
        [
            "",
            "## 5. Черновик формулировки выводов (для верификации экспертом)",
            "- Выявлен набор совпадающих признаков письменной речи, включая совпадающие формулировки, повторяющиеся n-граммы и частные коммуникативные маркеры.",
            "- Зафиксированы признаки, требующие экспертной интерпретации: синонимические замены, кандидаты на перефраз и случаи транслитерации.",
            "- Данный отчет носит аналитический вспомогательный характер и не заменяет итогового процессуального вывода эксперта.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Легкий MVP для автороведческого сравнения текстов")
    parser.add_argument("--query", required=True, type=Path, help="Путь к исследуемому тексту (.txt)")
    parser.add_argument("--sample", required=True, type=Path, help="Путь к образцу (.txt)")
    parser.add_argument("--report", default=Path("report.md"), type=Path, help="Куда сохранить отчет (.md)")
    args = parser.parse_args()

    query_text = args.query.read_text(encoding="utf-8")
    sample_text = args.sample.read_text(encoding="utf-8")
    result = run_analysis(query_text, sample_text)
    report = render_report(result, args.query.name, args.sample.name)
    args.report.write_text(report, encoding="utf-8")
    print(f"Готово. Отчет сохранен: {args.report}")


if __name__ == "__main__":
    run_cli()
