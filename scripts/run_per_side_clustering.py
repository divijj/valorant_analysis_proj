# exploratory: cluster agents using only attack-side or only defend-side stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    KILLS_STATS_CSV,
    MAPS_SCORES_CSV,
    OUTPUTS_DIR,
    OVERVIEW_CSV,
    RANDOM_SEED,
)
from vct_analysis.clustering import compute_kmeans_inertias, compute_linkage
from vct_analysis.data_loader import build_master_df
from vct_analysis.features import add_role_column, engineer_agent_features_side
from vct_analysis.visualization import plot_dendrogram, plot_elbow, plot_per_side_pca

MIN_MAPS = 50
K = 6
FEAT_COLS = [
    "entry_rate", "entry_success", "death_share",
    "kills_per_engagement", "assist_rate", "adr", "acs",
]


def run_side(master, side: str):
    print(f"\n{'=' * 60}\n  side: {side.upper()}\n{'=' * 60}")
    af = engineer_agent_features_side(master, side)
    af = af[af["n_maps"] >= MIN_MAPS].copy()

    print(f"\nagents retained (n_maps >= {MIN_MAPS}): {len(af)}")
    print(af[["agents", "role", "n_maps"]].sort_values("n_maps", ascending=False).to_string(index=False))

    x_scaled = StandardScaler().fit_transform(af[FEAT_COLS].values)

    linkage_matrix = compute_linkage(x_scaled)
    plot_dendrogram(
        linkage_matrix, af["agents"].values,
        save_path=OUTPUTS_DIR / f"dendrogram_{side}.png",
        title=f"agent dendrogram — {side} side",
    )

    inertias = compute_kmeans_inertias(x_scaled, range(2, 10))
    plot_elbow(
        inertias, range(2, 10),
        save_path=OUTPUTS_DIR / f"elbow_{side}.png",
        title=f"elbow — {side} side",
    )

    km = KMeans(n_clusters=K, random_state=RANDOM_SEED, n_init=10).fit(x_scaled)
    af["cluster"] = km.labels_

    print(f"\n=== k={K} centroids ({side}) ===")
    print(af.groupby("cluster")[FEAT_COLS].mean().round(3).T.to_string())

    print(f"\n=== cluster members ({side}) ===")
    for c in sorted(af["cluster"].unique()):
        members = af[af["cluster"] == c].sort_values("n_maps", ascending=False)
        print(f"  cluster_{c} (n={len(members)}): {', '.join(members['agents'])}")

    pca = PCA(n_components=2)
    coords = pca.fit_transform(x_scaled)
    plot_per_side_pca(af, pca, coords, K, side, save_path=OUTPUTS_DIR / f"pca_{side}.png")

    af.to_csv(OUTPUTS_DIR / f"agent_features_{side}.csv", index=False)


def main():
    master = build_master_df(OVERVIEW_CSV, KILLS_STATS_CSV, MAPS_SCORES_CSV)
    master = add_role_column(master)
    for side in ("attack", "defend","both"):
        run_side(master, side)


if __name__ == "__main__":
    main()