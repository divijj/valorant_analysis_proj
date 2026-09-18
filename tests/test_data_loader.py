# integration tests; skipped if raw csvs aren't present
import pytest

from config import KILLS_STATS_CSV, MAPS_SCORES_CSV, OVERVIEW_CSV
from vct_analysis.data_loader import build_master_df

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in (OVERVIEW_CSV, KILLS_STATS_CSV, MAPS_SCORES_CSV)),
    reason="raw csvs not present",
)


def test_master_no_row_inflation():
    # merge should never produce duplicate rows
    master = build_master_df(OVERVIEW_CSV, KILLS_STATS_CSV, MAPS_SCORES_CSV)
    keys = ["tournament", "stage", "match_type", "match_name", "map", "player", "side"]
    assert master.duplicated(subset=keys).sum() == 0


def test_win_rate_sanity():
    # across all maps, half the teams won, by definition
    master = build_master_df(OVERVIEW_CSV, KILLS_STATS_CSV, MAPS_SCORES_CSV)
    win_rate = master[master["side"] == "both"]["won"].mean()
    assert 0.45 < win_rate, f"win rate {win_rate} suspicious"


def test_kills_only_cols_null_on_per_side_rows():
    # kills_stats only exists at map level, per-side rows must have nan there
    master = build_master_df(OVERVIEW_CSV, KILLS_STATS_CSV, MAPS_SCORES_CSV)
    per_side = master[master["side"].isin(["attack", "defend"])]
    assert per_side["plants"].isna().all()
    assert per_side["k2"].isna().all()