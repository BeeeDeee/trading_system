"""SEC 8-K Item 2.02 dataset: parsing, resumable fetching, and the point-in-time reaction-day matrix."""

import json

import numpy as np

from lab.framework import attached, edgar

D = lambda s: np.datetime64(s, "D")  # noqa: E731
DATES = np.arange(D("2024-01-01"), D("2024-02-01"))
DATES = DATES[np.is_busday(DATES)]       # 2024-01-01 .. 01-31, Mon-Fri


def row(day):
    return int(np.searchsorted(DATES, D(day)))


def test_parse_keeps_primary_8k_with_item_202_only():
    block = {"form": ["8-K", "8-K/A", "10-Q", "8-K", "8-K"],
             "filingDate": ["2024-02-01", "2024-02-05", "2024-02-01", "2023-11-02", "2023-10-01"],
             "acceptanceDateTime": ["2024-02-01T21:05:00.000Z", "x", "y", "2023-11-02T12:00:00.000Z", "z"],
             "items": ["2.02,9.01", "2.02", "", "12.02,2.02", "5.02"]}
    first, rows = edgar.parse_block(block)
    assert first == "2023-10-01"
    assert rows == [("2024-02-01", "2024-02-01T21:05:00.000Z"), ("2023-11-02", "2023-11-02T12:00:00.000Z")]
    assert edgar.parse_block({}) == (None, [])


def test_fetch_cik_follows_older_files_and_unknown_cik_is_empty():
    recent = {"form": ["8-K"], "filingDate": ["2024-02-01"], "acceptanceDateTime": ["2024-02-01T21:00:00.000Z"],
              "items": ["2.02"]}
    older = {"form": ["8-K", "10-K"], "filingDate": ["2010-05-03", "2009-01-05"],
             "acceptanceDateTime": ["2010-05-03T11:00:00.000Z", "q"], "items": ["2.02", ""]}
    calls = []

    def get(url):
        calls.append(url)
        if url.endswith("CIK0000000005.json"):
            return json.dumps({"filings": {"recent": recent, "files": [{"name": "CIK0000000005-submissions-001.json"}]}}).encode()
        if url.endswith("-001.json"):
            return json.dumps(older).encode()
        return None
    got = edgar.fetch_cik(get, 5)
    assert got["first"] == "2009-01-05" and len(got["e202"]) == 2 and len(calls) == 2
    assert edgar.fetch_cik(get, 6) == {"first": None, "e202": []}


def test_fetch_all_resumes_from_cache(tmp_path):
    n = []

    def get(url):
        n.append(url)
        return json.dumps({"filings": {"recent": {}, "files": []}}).encode()
    edgar.fetch_all(get, [1, 2], tmp_path)
    edgar.fetch_all(get, [1, 2, 3], tmp_path)
    assert len(n) == 3


def test_reaction_session_is_the_close_at_or_after_acceptance():
    # Wed and Thu before the close (-> same day), Wed 17 after the close (-> Thu 18), Fri 19 after the close
    # (-> Mon 22), Sat 20 (-> Mon 22)
    day = np.array([D("2024-01-10"), D("2024-01-11"), D("2024-01-17"), D("2024-01-19"), D("2024-01-20")])
    after = np.array([False, False, True, True, False])
    m = attached.reaction_matrix(np.zeros(5, dtype=int), day, after, np.array([D("2000-01-01")]), DATES)
    ones = {str(DATES[i]) for i in np.flatnonzero(m[:, 0] == 1.0)}
    assert ones == {"2024-01-10", "2024-01-11", "2024-01-18", "2024-01-22"}
    assert m.dtype == np.float32 and (m[:, 0] == 0).sum() == len(DATES) - 4


def test_nan_before_the_first_filing_and_for_unknown_ciks():
    first = np.array([D("2024-01-15"), np.datetime64("NaT", "D")])
    m = attached.reaction_matrix(np.array([0]), np.array([D("2024-01-16")]), np.array([False]), first, DATES)
    assert np.isnan(m[: row("2024-01-15"), 0]).all() and m[row("2024-01-15"), 0] == 0.0
    assert m[row("2024-01-16"), 0] == 1.0 and np.isnan(m[:, 1]).all()


def test_filing_after_the_last_session_is_dropped():
    m = attached.reaction_matrix(np.array([0]), np.array([D("2024-01-31")]), np.array([True]),
                                 np.array([D("2020-01-01")]), DATES)
    assert (m == 0).all()


def test_the_field_is_validated_and_needs_sharadar_sep():
    cat = {"sec_8k_item202": {"loader": "sep_attached"}}
    card = {"data_requirements": [{"dataset": "sharadar_sep"}, {"dataset": "sec_8k_item202", "fields": ["ea_reaction_day"]}]}
    assert attached.requirement_problems(card, cat) == []
    card["data_requirements"][1]["fields"] = ["ea_day"]
    assert "not a field of sec_8k_item202" in attached.requirement_problems(card, cat)[0]


def test_user_agent_needs_a_contact(monkeypatch):
    monkeypatch.delenv("EDGAR_CONTACT", raising=False)
    try:
        edgar.user_agent()
        raise AssertionError("expected EdgarError")
    except edgar.EdgarError:
        pass
    monkeypatch.setenv("EDGAR_CONTACT", "a@b.example")
    assert edgar.user_agent() == "research-lab a@b.example"
