"""Archivist ingest: fetcher -> validation -> dev/holdout files -> catalog entry -> series in a DataView."""

from datetime import date, timedelta

import numpy as np
import pytest

from lab.framework import catalog, data, dryrun, ingest
from lab.framework.data import SYNTHETIC_DATASET

FETCHER = '''
import json

def fetch(get):
    body = json.loads(get("https://example.org/api", params={"start": "2015-01-01"}))
    return [(r["d"], r["k"], r["v"]) for r in body]
'''
DRAFT = {"id": "toy_vol_1d", "asset_class": "crypto_spot", "instruments": "toy implied vol index",
         "frequency": "1d", "clock": "next_morning", "source": "test", "known_biases": ["toy"],
         "forward_source": None}


def rows(start=date(2019, 1, 1), days=2500, keys=("BTC", "ETH")):
    return [{"d": str(start + timedelta(i)), "k": k, "v": 50 + (i % 30) + j}
            for i in range(days) for j, k in enumerate(keys)]


@pytest.fixture
def fake_http(monkeypatch):
    import json
    payload = {"rows": rows()}

    def get(self, url, params=None, headers=None):
        self.urls.append(url)
        return json.dumps(payload["rows"]).encode()
    monkeypatch.setattr(ingest.Http, "get", get)
    return payload


def test_validate_catches_bad_rows():
    with pytest.raises(ingest.IngestError, match="duplicate"):
        ingest.validate([("2020-01-01", "A", 1.0), ("2020-01-01", "A", 2.0), ("2020-01-02", "A", 1.0)])
    with pytest.raises(ingest.IngestError, match="future"):
        ingest.validate([("2020-01-01", "A", 1.0), ("2999-01-01", "A", 2.0)])
    with pytest.raises(ingest.IngestError, match="not \\(YYYY-MM-DD"):
        ingest.validate([("yesterday", "A", 1.0)])
    _, rep = ingest.validate([("2020-01-01", "A", 1.0), ("2020-01-15", "A", 2.0)])
    assert rep["keys"]["A"]["max_gap_days"] == 14


def test_boundary_uses_the_class_rule_then_70_percent():
    assert ingest.boundary("macro", date(1990, 1, 1), date(2026, 10, 1)) == date(2021, 1, 1)
    assert ingest.boundary("crypto_spot", date(2021, 3, 24), date(2026, 10, 1)) == date(2023, 1, 1)
    b = ingest.boundary("sentiment", date(2018, 1, 1), date(2026, 1, 1))
    assert date(2023, 1, 1) <= b <= date(2024, 1, 1)
    with pytest.raises(ingest.IngestError, match="too short"):
        ingest.boundary("alt", date(2025, 1, 1), date(2026, 1, 1))


def test_fetcher_static_rules_allow_dates_but_not_io():
    assert ingest.scan_fetcher(FETCHER) == []
    assert ingest.scan_fetcher("import os\ndef fetch(get):\n    return os.listdir('.')\n")
    assert ingest.scan_fetcher("def fetch(get):\n    return open('x').read()\n")


def test_draft_errors():
    assert ingest.draft_errors(DRAFT, "toy_vol_1d") == []
    bad = dict(DRAFT, frequency="1h", clock="whenever", known_biases=[])
    assert len(ingest.draft_errors(bad, "other_id")) == 4


def test_ingest_writes_split_files_and_a_generic_catalog_entry(tmp_path, fake_http):
    import shutil
    from lab.framework.paths import LAB_DIR
    cat = tmp_path / "catalog.yaml"
    shutil.copy(LAB_DIR / "data" / "catalog.yaml", cat)
    src = tmp_path / "toy_vol_1d.py"
    src.write_text(FETCHER)
    entry = ingest.ingest(tmp_path / "home", cat, "toy_vol_1d", dict(DRAFT, holdout_from="2019-06-01"), src)
    assert entry["holdout_from"] == "2023-01-01" and entry["loader"] == "generic"   # the framework's, not the draft's
    assert entry["fields"] == ["BTC", "ETH"] and entry["range"][0] == "2019-01-01"
    d = tmp_path / "home" / "data" / "toy_vol_1d"
    import polars as pl
    dev, hold = pl.read_parquet(d / "dev.parquet"), pl.read_parquet(d / "holdout.parquet")
    assert dev["date"].max() < date(2023, 1, 1) <= hold["date"].min()
    assert catalog.load(cat)["toy_vol_1d"]["loader"] == "generic"
    assert catalog.resolve(cat, [{"dataset": "toy_vol_1d", "frequency": "1d"}]).ok


def test_attach_generic_respects_the_clock(tmp_path, fake_http):
    fake_http["rows"] = [{"d": "2020-01-0%d" % i, "k": "X", "v": float(i)} for i in range(1, 8)]
    cat = tmp_path / "catalog.yaml"
    cat.write_text("version: 1\ndatasets: []\n")
    src = tmp_path / "s.py"
    src.write_text(FETCHER)
    ingest.fetch(src)   # smoke
    data_rows, _ = ingest.validate([(r["d"], r["k"], r["v"]) for r in fake_http["rows"]])
    import polars as pl
    out = tmp_path / "home" / "data" / "toy"
    out.mkdir(parents=True)
    ds, vs = data_rows["X"]
    pl.DataFrame({"date": ds, "key": ["X"] * len(ds), "value": vs}).write_parquet(out / "dev.parquet")
    view = data.synthetic().between(np.datetime64("2020-01-01"), np.datetime64("2020-01-10"))
    nm = data.attach_generic(view, tmp_path / "home", "toy", {"clock": "next_morning"})
    same = data.attach_generic(view, tmp_path / "home", "toy", {"clock": "us_close"})
    # 2020-01-02 (Thu): known at the close -> 2.0; published next morning -> still 1.0 from the day before
    i = int(np.flatnonzero(view.dates == np.datetime64("2020-01-02"))[0])
    assert same.series["toy.X"][i] == 2.0 and nm.series["toy.X"][i] == 1.0
    assert np.isnan(nm.series["toy.X"][0])          # 2020-01-01: nothing known yet with the lag


def test_lab_try_gives_signal_datasets_synthetic_series(card):
    cat = {"sharadar_sfp": {"range": ["1998-01-02", "2026-09-25"], "holdout_from": "2021-01-01",
                            "asset_class": "us_etf"},
           "toy_vol_1d": {"range": ["2010-01-01", "2026-09-01"], "holdout_from": "2021-01-01",
                          "asset_class": "macro", "fields": ["VIX"], "loader": "generic"}}
    card = dict(card, data_requirements=card["data_requirements"]
                + [{"dataset": "toy_vol_1d", "frequency": "1d", "fields": ["vix_close_guess"]}])
    v = dryrun.synthetic_view(card, cat, seed=3)
    assert list(v.series) == ["toy_vol_1d.VIX"]                     # the catalog's keys, not the card's guess
    assert "toy_vol_1d.VIX" in v.series and np.isnan(v.series["toy_vol_1d.VIX"][0])
    assert list(v.instruments) == card["universe"]["instruments"]


def test_synthetic_dataset_is_not_generic():
    assert SYNTHETIC_DATASET["loader"] is True
