import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from authorship_mvp import run_analysis


def test_analysis_finds_key_signals():
    query = "Я бегала сегодня утром и это было круууто!!! half марафон done"
    sample = "Я бегать сегодня утром начала. Это классно!! Половину марафона done"

    result = run_analysis(query, sample)

    assert result.query_stats.words > 0
    assert result.sample_stats.words > 0
    assert isinstance(result.match_result.ngram_matches, list)
    assert result.match_result.internet_markers["query_latin_tokens"] >= 1
