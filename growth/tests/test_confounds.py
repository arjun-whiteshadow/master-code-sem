"""Checks on the growth-log confound tests. Run with ``pytest`` from the growth folder."""

import numpy as np
import pandas as pd
import pytest

from src import confounds

SEED = 0


@pytest.fixture
def merged():
    """Thirty runs over 2023 with a temperature ramp, a flux tied to the
    temperature, an independent flux, and a substrate change mid-year."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2023-01-10", periods=30, freq="10D")
    ids = [f"{d:%m%d%y}B{i:02d}" for i, d in enumerate(dates)]
    T = np.linspace(500, 640, 30)
    return pd.DataFrame({
        "Sample_ID": ids,
        "Cluster": rng.integers(1, 4, 30),
        "T": T,
        "Ga": 2 * T + rng.normal(0, 5, 30),
        "As": rng.normal(1, 0.1, 30),
        "Substrate": ["Si"] * 15 + ["GaAs"] * 15,
    })


def test_growth_dates_parses_prefix_and_rejects_others():
    d = confounds.growth_dates(["070823B03", "111623DN06", "x", "133023A"])
    assert d["070823B03"] == pd.Timestamp("2023-07-08")
    assert d["111623DN06"] == pd.Timestamp("2023-11-16")
    assert pd.isna(d["x"]) and pd.isna(d["133023A"])


def test_parameter_correlations_flags_only_the_tied_pair(merged):
    c = confounds.parameter_correlations(merged, ["T", "Ga", "As"], 0.7)
    assert len(c) == 3
    c = c.set_index(["Parameter_A", "Parameter_B"])
    assert c.loc[("T", "Ga"), "Not_Separable"]
    assert not c.loc[("T", "As"), "Not_Separable"]
    assert not c.loc[("Ga", "As"), "Not_Separable"]


def test_parameter_correlations_uses_pairwise_complete_rows(merged):
    merged.loc[:4, "As"] = np.nan
    c = confounds.parameter_correlations(merged, ["T", "As"], 0.7)
    assert c.loc[0, "N"] == 25


def test_date_drift_flags_the_ramped_parameter(merged):
    dates = confounds.growth_dates(merged["Sample_ID"])
    d = confounds.date_drift(merged, ["T", "As"], dates, 0.7).set_index("Parameter")
    assert d.loc["T", "Drifts"] and d.loc["T", "Spearman_rho_vs_Date"] > 0.99
    assert not d.loc["As", "Drifts"]


def test_category_dates_gives_each_level_its_period(merged):
    dates = confounds.growth_dates(merged["Sample_ID"])
    t = confounds.category_dates(merged, ["Substrate"], dates).set_index("Level")
    assert t.loc["Si", "N"] == 15 and t.loc["GaAs", "N"] == 15
    assert t.loc["Si", "Last_Date"] < t.loc["GaAs", "First_Date"]
