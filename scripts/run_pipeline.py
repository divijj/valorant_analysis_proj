import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    AGENT_FEATURES_CSV,
    KILLS_STATS_CSV,
    K_DOWNSTREAM,
    K_VALUES,
    MAPS_SCORES_CSV,
    MASTER_CSV,
    MIN_MAPS_PER_AGENT,
    OUTPUTS_DIR,
    OVERVIEW_CSV,
    TEAMS_AGENTS_CSV,
    WIN_LOSS_CSV,
)
from vct_analysis.clustering import (
    build_agent_cluster_map,
    cluster_agents,
    compute_kmeans_inertias,
    compute_linkage,
    compute_silhouette_scores,
    get_feature_columns,
    print_cluster_centroids,
    print_cluster_members,
    print_feature_correlations,
    print_feature_diagnostics,
    print_pca_loadings,
    project_pca,
    scale_features,
)
from vct_analysis.composition import (
    cluster_diff_winners_vs_losers,
    compute_team_strength,
    find_optimal_cluster_comp,
    find_optimal_role_comp,
    load_composition_data,
    map_winrates,
    train_winrate_classifier,
)
from vct_analysis.data_loader import build_master_df
from vct_analysis.features import (
    add_role_column,
    engineer_agent_features,
    filter_low_sample_agents,
)
from vct_analysis.regression import build_regression_df, run_per_cluster_regression
from vct_analysis.visualization import (
    plot_cluster_winrate_heatmaps,
    plot_dendrogram,
    plot_elbow,
    plot_optimal_comp,
    plot_pca_scatter,
    plot_regression_coefs,
    plot_role_winrate_heatmaps,
    plot_silhouette,
    plot_winners_vs_losers_comp,
)


def main():
    # master dataframe in player-map format 
    print("\n>>> building master dataframe")
    master = build_master_df(OVERVIEW_CSV, KILLS_STATS_CSV, MAPS_SCORES_CSV)
    master = add_role_column(master)
    print(f"master shape: {master.shape}")
    print(f"win rate (sanity ~0.5): {master[master['side'] == 'both']['won'].mean():.3f}")

    # agent features (attack/defend)
    # the idea behind running the clustering and regression on attack and defense features individually 
    # is to identify patterns in how agents are played fundamentally, as their play on attack and defence would obviously differ
    print("\n>>> engineering agent features")
    agent_features = engineer_agent_features(master)
    master, agent_features, dropped = filter_low_sample_agents(
        master, agent_features, MIN_MAPS_PER_AGENT,
    )
    print(f"dropped agents (< {MIN_MAPS_PER_AGENT} maps): {dropped}")

    # dropping agents which have not recieved enough play as they would essentially be noise 
    feat_cols = get_feature_columns(agent_features)
    x_scaled = scale_features(agent_features, feat_cols)

    feat_cols = get_feature_columns(agent_features)
    x_scaled = scale_features(agent_features, feat_cols)

    print("\n>>> clustering feature correlations")
    print_feature_correlations(agent_features, feat_cols)

    # analysis + clustering at every k
    print("\n>>> clustering diagnostics")
    print_feature_diagnostics(agent_features, feat_cols, x_scaled)

    linkage_matrix = compute_linkage(x_scaled)
    # plot_dendrogram(
    #     linkage_matrix, agent_features["agents"].values,
    #     save_path=OUTPUTS_DIR / "dendrogram.png",
    # )

    inertias = compute_kmeans_inertias(x_scaled, range(2, 10))
    # plot_elbow(inertias, range(2, 10), save_path=OUTPUTS_DIR / "elbow.png")

    sil_scores = compute_silhouette_scores(x_scaled, range(2, 10))
    # plot_silhouette(sil_scores, range(2, 10), save_path=OUTPUTS_DIR / "silhouette.png")
    best_k = list(range(2, 10))[sil_scores.index(max(sil_scores))]
    print(f"\nsilhouette-recommended k: {best_k} (K_DOWNSTREAM currently set to {K_DOWNSTREAM})")

    agent_features = cluster_agents(agent_features, x_scaled, K_VALUES)
    for k in K_VALUES:
        print_cluster_members(agent_features, k)
        print_cluster_centroids(agent_features, k, feat_cols)

    # pca scatter
    pca, coords = project_pca(x_scaled)
    print_pca_loadings(pca, feat_cols)
    # plot_pca_scatter(
    #     agent_features, pca, coords, K_DOWNSTREAM,
    #     save_path=OUTPUTS_DIR / f"pca_k{K_DOWNSTREAM}.png",
    # )

    master.to_csv(MASTER_CSV, index=False)
    agent_features.to_csv(AGENT_FEATURES_CSV, index=False)
    print(f"\nsaved {MASTER_CSV.name}, {AGENT_FEATURES_CSV.name}")

    # per cluster regression
    print(f"\n>>> per-cluster regression at k={K_DOWNSTREAM}")
    reg = build_regression_df(master, agent_features, K_DOWNSTREAM)
    coef_table = run_per_cluster_regression(reg)
    coef_table.to_csv(OUTPUTS_DIR / f"regression_coefs_k{K_DOWNSTREAM}.csv")
    # plot_regression_coefs(
    #     coef_table, save_path=OUTPUTS_DIR / f"regression_coefs_k{K_DOWNSTREAM}.png",
    # )

    print(f"\n>>> per-cluster regression at k={K_DOWNSTREAM} (excluding kast)")
    coef_table_no_kast = run_per_cluster_regression(
        reg, predictors=["adr", "fkd", "multikill_score", "clutch_score"],
    )
    coef_table_no_kast.to_csv(OUTPUTS_DIR / f"regression_coefs_k{K_DOWNSTREAM}_no_kast.csv")
    # plot_regression_coefs(
    #     coef_table_no_kast, save_path=OUTPUTS_DIR / f"regression_coefs_k{K_DOWNSTREAM}_no_kast.png",
    # )

    # composition analysis
    print(f"\n>>> composition analysis at k={K_DOWNSTREAM}")
    agent_cluster_map = build_agent_cluster_map(agent_features, K_DOWNSTREAM)
    cluster_ids = sorted(set(agent_cluster_map.values()))

    comp_master = load_composition_data(TEAMS_AGENTS_CSV, WIN_LOSS_CSV, agent_cluster_map)
    print(f"composition rows: {comp_master.shape}")

    winrates = map_winrates(comp_master, cluster_ids)
    winrates.to_csv(OUTPUTS_DIR / "map_winrates.csv", index=False)
    print("\nmap winrates summary:")
    print(winrates.to_string(index=False))

    diff = cluster_diff_winners_vs_losers(comp_master, cluster_ids)
    print("\ncluster differential (winners - losers) per map:")
    print(diff.to_string(index=False))
    diff.to_csv(OUTPUTS_DIR / "cluster_diff_winners_vs_losers.csv", index=False)

    # plot_winners_vs_losers_comp(
    #     comp_master, cluster_ids, save_path=OUTPUTS_DIR / "winners_vs_losers.png",
    # )
    # plot_role_winrate_heatmaps(
    #     comp_master, save_path=OUTPUTS_DIR / "role_winrate_heatmaps.png",
    # )
    # if len(cluster_ids) >= 2:
        # demo cluster-vs-cluster heatmap with first two cluster ids
        # plot_cluster_winrate_heatmaps(
        #     comp_master, cluster_ids[0], cluster_ids[1],
        #     save_path=OUTPUTS_DIR / "cluster_winrate_heatmaps.png",
        # )

    optimal_roles = find_optimal_role_comp(comp_master)
    optimal_clusters = find_optimal_cluster_comp(comp_master, cluster_ids)
    print("\noptimal role composition per map:")
    print(optimal_roles.to_string(index=False))
    print("\noptimal cluster composition per map:")
    print(optimal_clusters.to_string(index=False))
    plot_optimal_comp(
        optimal_roles, optimal_clusters, cluster_ids,
        save_path=OUTPUTS_DIR / "optimal_comp_by_map.png",
    )
    optimal_roles.to_csv(OUTPUTS_DIR / "optimal_roles.csv", index=False)
    optimal_clusters.to_csv(OUTPUTS_DIR / "optimal_clusters.csv", index=False)

    # random forest baseline — composition only (team_avg_rating/acs dropped: same-match leakage)
    print("\n>>> random forest baseline")
    _, report, _, cv_scores = train_winrate_classifier(comp_master, cluster_ids)
    print(report)
    print("\n>>> pipeline complete")


if __name__ == "__main__":
    main()
    