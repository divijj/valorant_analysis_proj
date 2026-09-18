# build master player-map dataframe from overview + kills_stats + maps_scores
from pathlib import Path

import numpy as np
import pandas as pd

from vct_analysis.constants import (
    KEY,
    KILLS_ONLY_COLS,
    KILLS_STATS_RENAMES,
    MAPS_SCORES_RENAMES,
    OVERVIEW_RENAMES,
)


def audit_file(df: pd.DataFrame, name: str, key_cols: list) -> None:
    # sanity print: shape, duplicates on a key subset, null counts
    print(f"=== {name} ===")
    print(f"shape: {df.shape}")
    print(f"dupes on {key_cols}: {df[key_cols].duplicated().sum()}")
    print(f"nulls in keys: {df[key_cols].isna().sum().to_dict()}")
    print()


def _load_overview(path: Path) -> pd.DataFrame:
    ov = pd.read_csv(path).rename(columns=OVERVIEW_RENAMES)
    ov = ov[ov["map"] != "All Maps"].copy()
    # strip percent signs so we can do math
    ov["kast"] = ov["kast"].astype(str).str.replace("%", "", regex=False).replace("nan", np.nan).astype(float)
    ov["hs"] = ov["hs"].astype(str).str.replace("%", "", regex=False).replace("nan", np.nan).astype(float)
    assert ov.duplicated(subset=KEY + ["player", "side"]).sum() == 0
    return ov


def _load_kills_stats(path: Path) -> pd.DataFrame:
    ks = pd.read_csv(path).rename(columns=KILLS_STATS_RENAMES)
    ks = ks[ks["map"] != "All Maps"].copy()
    # missing multikill/clutch counts mean the player didn't get them, not unknown
    for c in KILLS_ONLY_COLS:
        ks[c] = ks[c].fillna(0)
    assert ks.duplicated(subset=KEY + ["player"]).sum() == 0
    return ks


def _load_winners(path: Path) -> pd.DataFrame:
    sc = pd.read_csv(path).rename(columns=MAPS_SCORES_RENAMES)
    assert sc.duplicated(subset=KEY).sum() == 0
    # stack team A and team B into long format — one row per (match, map, team)
    a = sc[KEY + ["team_a", "a_score", "b_score"]].rename(
        columns={"team_a": "team", "a_score": "score", "b_score": "opp_score"})
    b = sc[KEY + ["team_b", "b_score", "a_score"]].rename(
        columns={"team_b": "team", "b_score": "score", "a_score": "opp_score"})
    winners = pd.concat([a, b], ignore_index=True)
    winners["won"] = (winners["score"] > winners["opp_score"]).astype(int)
    assert (winners.groupby(KEY).size() == 2).all()
    return winners


def build_master_df(overview_path: Path, kills_path: Path, scores_path: Path) -> pd.DataFrame:
    # merges overview (player, map, side) + kills_stats (multikills/clutches) + winners.
    # output: one row per (player, map, side) with everything attached.
    ov = _load_overview(overview_path)
    ks = _load_kills_stats(kills_path)

    ks_cols = KEY + ["player", "team"] + KILLS_ONLY_COLS
    n0 = len(ov)
    master = ov.merge(ks[ks_cols], on=KEY + ["player", "team"], how="left")
    assert len(master) == n0, "kills_stats merge inflated row count"

    # kills_stats is per-map only; only attach to 'both'-side rows; null out attack/defend
    master.loc[master["side"] != "both", KILLS_ONLY_COLS] = np.nan

    n_missing = master[(master["side"] == "both") & (master["plants"].isna())].shape[0]
    if n_missing:
        print(f"warn: {n_missing} 'both' rows missing kills_stats")

    winners = _load_winners(scores_path)
    n0 = len(master)
    master = master.merge(
        winners[KEY + ["team", "won", "score", "opp_score"]],
        on=KEY + ["team"], how="left",
    )
    assert len(master) == n0, "winners merge inflated row count"
    if master["won"].isna().sum():
        print(f"warn: {master['won'].isna().sum()} rows missing 'won' (team-name mismatch)")

    return master


def load_or_build_master(
    overview_path: Path,
    kills_path: Path,
    scores_path: Path,
    cache_path: Path,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    # convenience; to skip the merge if we already cached master.csv
    if cache_path.exists() and not force_rebuild:
        print(f"loading cached master from {cache_path}")
        return pd.read_csv(cache_path)
    return build_master_df(overview_path, kills_path, scores_path)