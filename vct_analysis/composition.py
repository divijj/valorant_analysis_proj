# team composition analysis: how cluster/role mix affects win rate per map
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split  
from sklearn.preprocessing import LabelEncoder

from config import MIN_HEATMAP_SAMPLES, RANDOM_SEED
from vct_analysis.features import get_role


def load_composition_data(teams_agents_path, win_loss_path, agent_cluster_map: dict) -> pd.DataFrame:
    # returns one row per (map, team) with cluster counts, role counts, and won flag
    comp = pd.read_csv(teams_agents_path)
    map_df = pd.read_csv(win_loss_path)

    comp.columns = comp.columns.str.strip().str.replace(" ", "_").str.lower()
    map_df.columns = map_df.columns.str.strip().str.replace(" ", "_").str.lower()

    comp = comp[
        (comp["stage"] != "All Stages")
        & (comp["match_type"] != "All Match Types")
        & (comp["map"] != "All Maps")
    ]

    # make agent strings uniformly lowercase
    comp["agent"] = comp["agent"].astype(str).str.strip().str.lower()

    # unmapped agents: those filtered out for low sample size)
    comp["cluster_raw"] = comp["agent"].map(agent_cluster_map)
    unmapped = sorted(set(comp.loc[comp["cluster_raw"].isna(), "agent"]))
    if unmapped:
        print(f"warn: {len(unmapped)} agents not in cluster map (excluded from cluster counts): {unmapped}")
    comp["cluster_raw"] = comp["cluster_raw"].fillna(-1).astype(int)
    comp["role"] = comp["agent"].apply(get_role)

    # only positive cluster ids get count columns
    cluster_ids = sorted({c for c in agent_cluster_map.values()})
    agg_dict = {
        f"cluster_{c}_count": ("cluster_raw", lambda x, c=c: (x == c).sum())
        for c in cluster_ids
    }
    for r in ("duelist", "initiator", "controller", "sentinel"):
        agg_dict[f"{r}_count"] = ("role", lambda x, r=r: (x == r).sum())

    team_comp = comp.groupby(
        ["tournament", "stage", "match_type", "map", "team"],
        as_index=False,
    ).agg(**agg_dict)

    # derive won/loss from method counts; one row per (team, map) here
    map_df["total_wins"] = map_df["elimination"] + map_df["detonated"] + map_df["defused"]
    map_df["total_losses"] = (
        map_df["eliminated"]
        + map_df["defused_failed"]
        + map_df["detonation_denied"]
        + map_df["time_expiry_(failed_to_plant)"]
    )
    map_df["won"] = (map_df["total_wins"] > map_df["total_losses"]).astype(int)

    master = team_comp.merge(
        map_df[
            [
                "tournament", "stage", "match_type", "map", "team",
                "total_wins", "total_losses", "won",
            ]
        ],
        on=["tournament", "stage", "match_type", "map", "team"],
        how="left",
    )
    return master


def map_winrates(master: pd.DataFrame, cluster_ids: list) -> pd.DataFrame:
    cluster_avg = {f"avg_cluster_{c}": (f"cluster_{c}_count", "mean") for c in cluster_ids}
    role_avg = {
        f"avg_{r}": (f"{r}_count", "mean")
        for r in ("duelist", "initiator", "controller", "sentinel")
    }
    return master.groupby("map", as_index=False).agg(
        total_maps=("won", "count"),
        win_rate=("won", "mean"),
        **cluster_avg,
        **role_avg,
    )


def cluster_diff_winners_vs_losers(master: pd.DataFrame, cluster_ids: list) -> pd.DataFrame:
    # per-map difference in avg cluster counts between winning and losing teams
    cluster_cols = [f"cluster_{c}_count" for c in cluster_ids]
    rows = []
    for map_name in sorted(master["map"].unique()):
        map_data = master[master["map"] == map_name]
        winners = map_data[map_data["won"] == 1][cluster_cols].mean()
        losers = map_data[map_data["won"] == 0][cluster_cols].mean()
        row = {"map": map_name, **(winners - losers).to_dict()}
        rows.append(row)
    return pd.DataFrame(rows)


def find_optimal_role_comp(master: pd.DataFrame, min_samples: int = MIN_HEATMAP_SAMPLES) -> pd.DataFrame:
    role_cols = ["duelist_count", "initiator_count", "controller_count", "sentinel_count"]
    heatmap = master.groupby(["map"] + role_cols)["won"].agg(
        win_rate="mean", sample_size="count",
    ).reset_index()
    heatmap = heatmap[heatmap["sample_size"] >= min_samples]
    return heatmap.loc[heatmap.groupby("map")["win_rate"].idxmax()].reset_index(drop=True)


def find_optimal_cluster_comp(
    master: pd.DataFrame, cluster_ids: list, min_samples: int = MIN_HEATMAP_SAMPLES,
) -> pd.DataFrame:
    cluster_cols = [f"cluster_{c}_count" for c in cluster_ids]
    heatmap = master.groupby(["map"] + cluster_cols)["won"].agg(
        win_rate="mean", sample_size="count",
    ).reset_index()
    heatmap = heatmap[heatmap["sample_size"] >= min_samples]
    return heatmap.loc[heatmap.groupby("map")["win_rate"].idxmax()].reset_index(drop=True)


def train_winrate_classifier(
    master: pd.DataFrame,
    cluster_ids: list,
    extra_features: list = None,
    cv_folds: int = 5,
):
    le = LabelEncoder()
    df = master.copy()
    df["map_encoded"] = le.fit_transform(df["map"])
    features = (
        ["map_encoded"]
        + [f"cluster_{c}_count" for c in cluster_ids]
        + ["duelist_count", "initiator_count", "controller_count", "sentinel_count"]
        + (extra_features or [])
    )
    df = df.dropna(subset=features + ["won"])
    x = df[features]
    y = df["won"]

    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
        x, y, cv=cv_folds,
    )
    print(f"\n{cv_folds}-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_SEED,
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(x_train, y_train)
    report = classification_report(y_test, rf.predict(x_test))
    return rf, report, features, cv_scores


def compute_team_strength(master: pd.DataFrame) -> pd.DataFrame:
    # avg rating per (match key, team) from 'both' side rows; a team-skill prior for the RF
    key = ["tournament", "stage", "match_type", "map", "team"]
    both = master[master["side"] == "both"]
    return both.groupby(key, as_index=False).agg(
        team_avg_rating=("rating", "mean"),
        team_avg_acs=("acs", "mean"),
    )
