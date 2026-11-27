"""Tests matchmaking (V3 §10): αμοιβαίο/μονόπλευρο ζεύγος, κενά needs, ισοβαθμία, self-match, άσχετο.

Τρέχουν πάνω στο datasets/mini_profiles.json (8 προφίλ με γνωστές αναμενόμενες τιμές).
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "spark"))

from data_loader import load_profiles  # noqa: E402
from matchmaking import compute_partners  # noqa: E402
from spark_config import create_spark_session  # noqa: E402

MINI = os.path.join(ROOT, "datasets", "mini_profiles.json")
SHARED_SAMPLE = os.path.join(ROOT, "tests", "fixtures", "similarity_sample.json")


@pytest.fixture(scope="module")
def spark():
    s = create_spark_session("test-matchmaking")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def result(spark):
    profiles = load_profiles(spark, MINI)
    rows = compute_partners(profiles).collect()
    out = {}
    for r in rows:
        out.setdefault(r.profile_id, []).append(r)
    return out


def test_shared_sample_runs_without_self_match(spark):
    """Smoke test πάνω στο κοινό fixture της Μιλένας (5 προφίλ)."""
    rows = compute_partners(load_profiles(spark, SHARED_SAMPLE)).collect()
    for r in rows:
        assert r.profile_id != r.partner_id
        assert 0.0 <= r.match_score <= 1.0
        assert r.match_reason


def test_terms_are_normalized(spark, tmp_path):
    """Διαφορετικά κεφαλαία/κενά στον ίδιο όρο πρέπει να ταιριάζουν (ο loader δεν κανονικοποιεί)."""
    import json

    data = [
        {"id": 1, "name": "A", "category": "Technology", "location": "Patras", "needs": ["  Digital Marketing "]},
        {"id": 2, "name": "B", "category": "Marketing", "location": "Patras", "services": ["digital marketing"]},
    ]
    p = tmp_path / "norm.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    rows = compute_partners(load_profiles(spark, p)).collect()
    assert {(r.profile_id, r.partner_id) for r in rows} == {(1, 2), (2, 1)}


def partners(result, pid):
    return [(r.partner_id, r.match_score) for r in result.get(pid, [])]


def test_mutual_pair_gets_full_score(result):
    # 1 χρειάζεται Digital Marketing (το προσφέρει το 2) και το 2 χρειάζεται Web Development (το προσφέρει το 1)
    assert partners(result, 1)[0] == (2, 1.0)


def test_one_sided_pair_scores_lower(result):
    # 3 προσφέρει Digital Marketing αλλά δεν έχει ανάγκες: forward 1, reverse 0, κλάδος 1, περιοχή 0
    assert (3, 0.6) in partners(result, 1)
    assert partners(result, 1)[0][1] > partners(result, 1)[1][1]


def test_empty_needs_still_matches_via_reverse(result):
    # Το 3 δεν έχει needs, αλλά οι 1,4,7 χρειάζονται αυτό που προσφέρει
    ids = [p for p, _ in partners(result, 3)]
    assert set(ids) == {1, 4, 7}
    assert partners(result, 3)[0] == (4, 0.5)  # ίδια περιοχή + συμβατός κλάδος


def test_tie_break_by_partner_id(result):
    # Τα 1 και 7 έχουν ακριβώς το ίδιο score για το 2 → μικρότερο id πρώτο
    assert partners(result, 2)[:2] == [(1, 1.0), (7, 1.0)]


def test_no_self_match(result):
    for pid, rows in result.items():
        assert pid not in [r.partner_id for r in rows]


def test_unrelated_and_empty_profiles_get_nothing(result):
    assert 6 not in result  # Agro Farm: κανένα need, κανείς δεν το χρειάζεται
    assert 8 not in result  # προφίλ χωρίς πληροφορίες


def test_category_or_location_alone_is_not_enough(result):
    # Το 6 είναι Patras όπως τα 1,2,7 αλλά δεν υπάρχει κάλυψη ανάγκης → δεν προτείνεται ποτέ
    for rows in result.values():
        assert 6 not in [r.partner_id for r in rows]


def test_scores_in_range_and_ranks_sequential(result):
    for rows in result.values():
        assert [r.rank_position for r in rows] == list(range(1, len(rows) + 1))
        for r in rows:
            assert 0.0 <= r.match_score <= 1.0
            assert len(r.matched_skills) >= 0
            assert r.match_reason


def test_reason_mentions_concrete_terms(result):
    r = result[1][0]
    assert "digital marketing" in r.match_reason.lower()
    assert list(r.matched_skills) == ["digital marketing"]
