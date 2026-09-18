# plotting helpers — each function takes an optional save_path
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram


def _save_and_show(fig, save_path):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_dendrogram(linkage_matrix, labels, save_path: Path = None, title: str = None) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    ax.set_title(title or "agent playstyle dendrogram (ward linkage)")
    ax.set_ylabel("distance")
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_elbow(inertias, k_range, save_path: Path = None, title: str = "elbow method") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(k_range), inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title(title)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_pca_scatter(
    agent_features: pd.DataFrame,
    pca,
    coords,
    k: int,
    sample_col: str = "n_player_maps",
    low_n_threshold: int = 50,
    save_path: Path = None,
) -> None:
    # marker size scaled by sample size; italic labels for low-n agents
    cluster_col = f"cluster_k{k}"
    df = agent_features.copy()
    df["pca1"] = coords[:, 0]
    df["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for c in range(k):
        sub = df[df[cluster_col] == c]
        sizes = 30 + 80 * np.log1p(sub[sample_col]) / np.log1p(df[sample_col]).max()
        ax.scatter(
            sub["pca1"], sub["pca2"], c=[colors[c]], s=sizes,
            label=f"cluster_{c} (n={len(sub)})",
            alpha=0.8, edgecolors="black", linewidths=0.5,
        )

    for _, row in df.iterrows():
        style = "italic" if row[sample_col] < low_n_threshold else "normal"
        weight = "normal" if row[sample_col] < low_n_threshold else "bold"
        ax.annotate(
            row["agents"], (row["pca1"], row["pca2"]),
            fontsize=9, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points",
            style=style, fontweight=weight,
        )

    ax.set_xlabel(f"pc1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"pc2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"agent playstyle clusters (k={k})\nmarker size proportional to sample size")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_regression_coefs(coef_table: pd.DataFrame, save_path: Path = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    coef_table.plot(kind="bar", ax=ax, width=0.85)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("standardized log-odds coefficient")
    ax.set_xlabel("predictor")
    ax.set_title("what predicts winning, per cluster (team fixed effects, n>=100 only)")
    ax.legend(title="cluster", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_winners_vs_losers_comp(master: pd.DataFrame, cluster_ids: list, save_path: Path = None) -> None:
    # one subplot per map: bar pair of (winners avg cluster count, losers avg cluster count)
    cluster_cols = [f"cluster_{c}_count" for c in cluster_ids]
    labels = [f"cluster_{c}" for c in cluster_ids]
    maps = sorted(master["map"].unique())
    ncols = 4
    nrows = (len(maps) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, map_name in enumerate(maps):
        map_data = master[master["map"] == map_name]
        winners = map_data[map_data["won"] == 1][cluster_cols].mean()
        losers = map_data[map_data["won"] == 0][cluster_cols].mean()
        x = np.arange(len(cluster_cols))
        w = 0.35
        axes[idx].bar(x - w / 2, winners, w, label="winners", color="steelblue")
        axes[idx].bar(x + w / 2, losers, w, label="losers", color="crimson")
        axes[idx].set_title(map_name)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        axes[idx].legend(fontsize=7)
        axes[idx].set_ylabel("avg agents")

    for idx in range(len(maps), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("winning vs losing team cluster composition by map", fontsize=14)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_role_winrate_heatmaps(master: pd.DataFrame, min_samples: int = 5, save_path: Path = None) -> None:
    # win rate cells across (controller_count, initiator_count) per map 
    maps = sorted(master["map"].unique())
    ncols = 4
    nrows = (len(maps) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, map_name in enumerate(maps):
        map_data = master[master["map"] == map_name]
        pivot = map_data.groupby(["controller_count", "initiator_count"])["won"].agg(
            win_rate="mean", n="count",
        ).reset_index()
        pivot = pivot[pivot["n"] >= min_samples]
        pivot_table = pivot.pivot(
            index="controller_count", columns="initiator_count", values="win_rate",
        )
        sns.heatmap(
            pivot_table, ax=axes[idx], annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=0.3, vmax=0.7, linewidths=0.5,
        )
        axes[idx].set_title(map_name)
        axes[idx].set_xlabel("initiator count")
        axes[idx].set_ylabel("controller count")

    for idx in range(len(maps), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("win rate by controller vs initiator count per map", fontsize=14)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_cluster_winrate_heatmaps(
    master: pd.DataFrame,
    cluster_a: int,
    cluster_b: int,
    min_samples: int = 5,
    save_path: Path = None,
) -> None:
    col_a = f"cluster_{cluster_a}_count"
    col_b = f"cluster_{cluster_b}_count"
    maps = sorted(master["map"].unique())
    ncols = 4
    nrows = (len(maps) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, map_name in enumerate(maps):
        map_data = master[master["map"] == map_name]
        pivot = map_data.groupby([col_a, col_b])["won"].agg(
            win_rate="mean", n="count",
        ).reset_index()
        pivot = pivot[pivot["n"] >= min_samples]
        pivot_table = pivot.pivot(index=col_a, columns=col_b, values="win_rate")
        sns.heatmap(
            pivot_table, ax=axes[idx], annot=True, fmt=".2f",
            cmap="RdYlGn", vmin=0.3, vmax=0.7, linewidths=0.5,
        )
        axes[idx].set_title(map_name)
        axes[idx].set_xlabel(f"cluster_{cluster_b} count")
        axes[idx].set_ylabel(f"cluster_{cluster_a} count")

    for idx in range(len(maps), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"win rate by cluster_{cluster_a} vs cluster_{cluster_b} count per map", fontsize=14)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_optimal_comp(
    optimal_roles: pd.DataFrame,
    optimal_clusters: pd.DataFrame,
    cluster_ids: list,
    save_path: Path = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    role_plot = optimal_roles.set_index("map")[
        ["duelist_count", "initiator_count", "controller_count", "sentinel_count"]
    ]
    sns.heatmap(role_plot, ax=axes[0], annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, cbar=False)
    axes[0].set_title("optimal role composition per map", fontsize=13)
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(["duelist", "initiator", "controller", "sentinel"])

    cluster_cols = [f"cluster_{c}_count" for c in cluster_ids]
    cluster_labels = [f"cluster_{c}" for c in cluster_ids]
    cluster_plot = optimal_clusters.set_index("map")[cluster_cols]
    sns.heatmap(cluster_plot, ax=axes[1], annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, cbar=False)
    axes[1].set_title("optimal cluster composition per map", fontsize=13)
    axes[1].set_xlabel("")
    axes[1].set_xticklabels(cluster_labels)

    plt.suptitle("optimal team composition by map — vct 2026", fontsize=15, y=1.02)
    plt.tight_layout()
    _save_and_show(fig, save_path)


def plot_per_side_pca(
    af: pd.DataFrame, pca, coords, k: int, side: str, save_path: Path = None,
) -> None:
    df = af.copy()
    df["pca1"] = coords[:, 0]
    df["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for c in range(k):
        sub = df[df["cluster"] == c]
        sizes = 30 + 80 * np.log1p(sub["n_maps"]) / np.log1p(df["n_maps"]).max()
        ax.scatter(
            sub["pca1"], sub["pca2"], c=[colors[c]], s=sizes,
            label=f"cluster_{c} (n={len(sub)})",
            alpha=0.8, edgecolors="black", linewidths=0.5,
        )
    for _, row in df.iterrows():
        ax.annotate(
            row["agents"], (row["pca1"], row["pca2"]),
            fontsize=9, ha="left", va="bottom",
            xytext=(4, 4), textcoords="offset points", fontweight="bold",
        )
    ax.set_xlabel(f"pc1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"pc2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(f"agent playstyle clusters — {side} side (k={k})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig, save_path)

def plot_silhouette(scores: list, k_range, save_path: Path = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    k_list = list(k_range)
    ax.plot(k_list, scores, marker="o")
    ax.axvline(k_list[int(np.argmax(scores))], color="crimson", linestyle="--", alpha=0.6,
               label=f"best k={k_list[int(np.argmax(scores))]}")
    ax.set_xlabel("k")
    ax.set_ylabel("silhouette score")
    ax.set_title("silhouette score by k (higher = better-separated clusters)")
    ax.legend()
    plt.tight_layout()
    _save_and_show(fig, save_path)