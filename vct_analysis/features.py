# role labels + agent feature engineering for clustering
import numpy as np
import pandas as pd

from vct_analysis.constants import CONTROLLER, DUELIST, INITIATOR, SENTINEL


def get_role(agent):
    # multi-agent rows (played more than one agent over a series) get NaN as can't pick a single role
    if pd.isna(agent) or "," in str(agent):
        return np.nan
    a = str(agent).strip().lower()
    if a in DUELIST:
        return "duelist"
    if a in SENTINEL:
        return "sentinel"
    if a in INITIATOR:
        return "initiator"
    if a in CONTROLLER:
        return "controller"
    return np.nan


def add_role_column(master: pd.DataFrame) -> pd.DataFrame:
    master = master.copy()
    master["role"] = master["agents"].apply(get_role)
    return master


def engineer_agent_features(master: pd.DataFrame) -> pd.DataFrame:
    # two-step aggregation:
    #   (player, agent, side) totals  ->  (agent, side) averages
    # equal-weighting players keeps high-volume pros from dominating an agent's profile.
    # final shape: one row per agent with atk_ and def_ columns side-by-side.
    df = master[master["role"].notna() & master["side"].isin(["attack", "defend"])].copy()

    pas = df.groupby(["player", "agents", "side"], as_index=False).agg(
        kills=("kills", "sum"),
        deaths=("deaths", "sum"),
        assists=("assists", "sum"),
        fk=("fk", "sum"),
        fd=("fd", "sum"),
        adr=("adr", "mean"),
        acs=("acs", "mean"),
        n_maps=("map", "count"),
    )

    asd = pas.groupby(["agents", "side"], as_index=False).agg(
        kills=("kills", "sum"),
        deaths=("deaths", "sum"),
        assists=("assists", "sum"),
        fk=("fk", "sum"),
        fd=("fd", "sum"),
        adr=("adr", "mean"),
        acs=("acs", "mean"),
        n_maps=("n_maps", "sum"),
        n_players=("player", "nunique"),
    )

    # engineered ratios — sum-then-divide for stability against zero denominators
    asd["entry_rate"] = (asd["fk"] + asd["fd"]) / (asd["kills"] + asd["deaths"])
    asd["entry_success"] = asd["fk"] / (asd["fk"] + asd["fd"]).replace(0, np.nan)
    asd["death_share"] = asd["deaths"] / (asd["kills"] + asd["deaths"])
    asd["kills_per_engagement"] = asd["kills"] / (asd["kills"] + asd["deaths"])
    asd["assist_rate"] = asd["assists"] / (asd["kills"] + asd["deaths"] + asd["assists"])

    feat_cols = ["entry_rate", "entry_success", "kills_per_engagement", "assist_rate", "adr", "acs"]
    wide = asd.pivot(index="agents", columns="side", values=feat_cols)
    wide.columns = [f"{'atk' if side == 'attack' else 'def'}_{feat}" for feat, side in wide.columns]
    wide = wide.reset_index()

    role_map = master[["agents", "role"]].dropna().drop_duplicates()
    wide = wide.merge(role_map, on="agents", how="left")

    n_total = (
        asd.groupby("agents")["n_maps"]
        .sum()
        .reset_index()
        .rename(columns={"n_maps": "n_player_maps"})
    )
    wide = wide.merge(n_total, on="agents", how="left")
    return wide


def engineer_agent_features_side(master: pd.DataFrame, side: str) -> pd.DataFrame:
    # one row per agent, features built from a single side only
    assert side in ("attack", "defend")
    df = master[master["role"].notna() & (master["side"] == side)].copy()

    pas = df.groupby(["player", "agents"], as_index=False).agg(
        kills=("kills", "sum"),
        deaths=("deaths", "sum"),
        assists=("assists", "sum"),
        fk=("fk", "sum"),
        fd=("fd", "sum"),
        adr=("adr", "mean"),
        acs=("acs", "mean"),
        n_maps=("map", "count"),
    )

    asd = pas.groupby("agents", as_index=False).agg(
        kills=("kills", "sum"),
        deaths=("deaths", "sum"),
        assists=("assists", "sum"),
        fk=("fk", "sum"),
        fd=("fd", "sum"),
        adr=("adr", "mean"),
        acs=("acs", "mean"),
        n_maps=("n_maps", "sum"),
        n_players=("player", "nunique"),
    )

    asd["entry_rate"] = (asd["fk"] + asd["fd"]) / (asd["kills"] + asd["deaths"])
    asd["entry_success"] = asd["fk"] / (asd["fk"] + asd["fd"]).replace(0, np.nan)
    asd["death_share"] = asd["deaths"] / (asd["kills"] + asd["deaths"])
    asd["kills_per_engagement"] = asd["kills"] / (asd["kills"] + asd["deaths"])
    asd["assist_rate"] = asd["assists"] / (asd["kills"] + asd["deaths"] + asd["assists"])

    role_map = master[["agents", "role"]].dropna().drop_duplicates()
    asd = asd.merge(role_map, on="agents", how="left")
    asd["side"] = side
    return asd


def filter_low_sample_agents(
    master: pd.DataFrame,
    agent_features: pd.DataFrame,
    min_maps: int,
    sample_col: str = "n_player_maps",
):
    # drop agents with too few player-maps from both feature table and master
    dropped = set(agent_features.loc[agent_features[sample_col] < min_maps, "agents"])
    agent_features = agent_features[agent_features[sample_col] >= min_maps].copy()
    master = master[~master["agents"].isin(dropped)].copy()
    return master, agent_features, dropped