# unit tests for role assignment, no csvs required
import numpy as np
import pandas as pd

from vct_analysis.features import get_role


def test_role_assignment_known_agents():
    assert get_role("jett") == "duelist"
    assert get_role("cypher") == "sentinel"
    assert get_role("sova") == "initiator"
    assert get_role("omen") == "controller"


def test_role_assignment_case_insensitive():
    assert get_role("Jett") == "duelist"
    assert get_role("  OMEN  ") == "controller"


def test_role_multi_agent_is_nan():
    # rows where a player swapped agents within the series stay unassigned
    assert pd.isna(get_role("jett,reyna"))


def test_role_unknown_agent_is_nan():
    assert pd.isna(get_role("notanagent"))


def test_role_nan_input():
    assert pd.isna(get_role(np.nan))