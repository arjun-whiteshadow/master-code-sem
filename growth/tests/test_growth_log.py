"""Checks on growth-log loading and joining. Run with ``pytest`` from the growth folder."""

import pandas as pd
import pytest

from src import growth_log

CFG = {"parameters": {"numeric": ["T", "Flux"], "categorical": ["Substrate"],
                      "min_valid_fraction": 0.8}}


@pytest.fixture
def log():
    return pd.DataFrame({
        "Sample_ID": [f"S{i}" for i in range(10)],
        "T": [500, 520, 540, 560, 580, 600, 620, 640, 660, 680],
        "Flux": [1.0] * 10,
        "Substrate": ["Si"] * 5 + ["GaAs"] * 5,
        "Notes": [""] * 9 + ["x"],
    })


def test_load_growth_log_skips_blank_rows_and_rejects_duplicates(tmp_path):
    p = tmp_path / "log.csv"
    cfg = {"log_path": p, "input": {"sample_id_column": "Sample"}}
    p.write_text("Sample,T\nA,1\n,\nB,2\n")
    df = growth_log.load_growth_log(cfg)
    assert df["Sample_ID"].tolist() == ["A", "B"]

    p.write_text("Sample,T\nA,1\nA,2\n")
    with pytest.raises(SystemExit):
        growth_log.load_growth_log(cfg)


def test_screen_parameters_drops_constant_and_reports_unlisted(log):
    numeric, categorical, audit = growth_log.screen_parameters(CFG, log)
    assert numeric == ["T"] and categorical == ["Substrate"]
    a = audit.set_index("Parameter")
    assert a.loc["Flux", "Status"] == "excluded"
    assert "constant" in a.loc["Flux", "Reason"]
    assert a.loc["Notes", "Status"] == "not used"


def test_screen_parameters_applies_valid_fraction(log):
    log.loc[:2, "T"] = None                        # 7 of 10 valid
    numeric, _, audit = growth_log.screen_parameters(CFG, log)
    assert numeric == []
    assert audit.set_index("Parameter").loc["T", "Status"] == "excluded"


def test_screen_parameters_reports_missing_column(log):
    _, _, audit = growth_log.screen_parameters(CFG, log.drop(columns="Flux"))
    assert audit.set_index("Parameter").loc["Flux", "Status"] == "missing"


def test_screen_parameters_rejects_text_in_numeric_column(log):
    log["T"] = log["T"].astype(object)
    log.loc[0, "T"] = "550 C"
    with pytest.raises(SystemExit):
        growth_log.screen_parameters(CFG, log)


def test_merge_reports_samples_missing_from_either_side(log):
    ids = [f"S{i}" for i in range(2, 10)] + ["S99"]
    X = pd.DataFrame({"d1": range(9), "d2": range(9)},
                     index=pd.Index(ids, name="Sample_ID"))
    clusters = pd.Series([1, 2] * 4 + [1], index=X.index)
    merged, coverage = growth_log.merge_with_method1(log, X, clusters)
    assert len(merged) == 8
    assert list(merged.columns[:2]) == ["Sample_ID", "Cluster"]
    cov = coverage.set_index("Sample_ID")
    assert cov.loc["S0", "In_Growth_Log"] and not cov.loc["S0", "In_Method1"]
    assert cov.loc["S99", "In_Method1"] and not cov.loc["S99", "In_Growth_Log"]
    assert int(cov["Included"].sum()) == 8


def test_merge_fails_when_nothing_matches(log):
    X = pd.DataFrame({"d1": [0.0]}, index=pd.Index(["other"], name="Sample_ID"))
    with pytest.raises(SystemExit):
        growth_log.merge_with_method1(log, X, pd.Series([1], index=X.index))
