# kmeans clustering + pca projection + hierarchical diagnostic
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from config import RANDOM_SEED


def get_feature_columns(agent_features: pd.DataFrame, exclude_suffixes: tuple = ("acs",)) -> list:
    # wide-format features are prefixed atk_ or def_
    # acs excluded by default as it is a composite of kills/adr/etc
    cols = [c for c in agent_features.columns if c.startswith(("atk_", "def_"))]
    return [c for c in cols if not any(c.endswith(f"_{s}") for s in exclude_suffixes)]


def scale_features(agent_features: pd.DataFrame, feat_cols: list) -> np.ndarray:
    return StandardScaler().fit_transform(agent_features[feat_cols].values)


def print_feature_diagnostics(agent_features: pd.DataFrame, feat_cols: list, x_scaled: np.ndarray) -> None:
    # variance per feature + pairwise distance; checks features actually have signal
    print("feature std (raw):")
    print(agent_features[feat_cols].std().round(3).sort_values(ascending=False).to_string())
    print(f"\nmean pairwise distance (z-scored): {pdist(x_scaled).mean():.2f}")
    print(f"max pairwise distance (z-scored):  {pdist(x_scaled).max():.2f}")


def compute_linkage(x_scaled: np.ndarray) -> np.ndarray:
    return linkage(x_scaled, method="ward")


def compute_kmeans_inertias(x_scaled: np.ndarray, k_range) -> list:
    return [
        KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(x_scaled).inertia_
        for k in k_range
    ]


def cluster_agents(agent_features: pd.DataFrame, x_scaled: np.ndarray, k_values: list) -> pd.DataFrame:
    # assigns one cluster column per k value; drops any stale cluster_k* from prior runs
    out = agent_features.copy()
    for col in [c for c in out.columns if c.startswith("cluster_k")]:
        out = out.drop(columns=col)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(x_scaled)
        out[f"cluster_k{k}"] = km.labels_
    return out


def print_cluster_members(agent_features: pd.DataFrame, k: int, sample_col: str = "n_player_maps") -> None:
    col = f"cluster_k{k}"
    print(f"\n=== k={k} cluster members ===")
    for c in sorted(agent_features[col].unique()):
        members = agent_features[agent_features[col] == c].sort_values(sample_col, ascending=False)
        print(f"  cluster_{c} (n={len(members)}): {', '.join(members['agents'])}")


def print_cluster_centroids(agent_features: pd.DataFrame, k: int, feat_cols: list) -> None:
    col = f"cluster_k{k}"
    print(f"\n=== k={k} centroids (raw) ===")
    print(agent_features.groupby(col)[feat_cols].mean().round(3).T.to_string())


def project_pca(x_scaled: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(x_scaled)
    return pca, coords


def print_pca_loadings(pca: PCA, feat_cols: list) -> None:
    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=feat_cols)
    print("top contributors to pc1:")
    print(loadings.abs().sort_values("PC1", ascending=False).head(6))
    print(
        f"\nvariance explained: pc1={pca.explained_variance_ratio_[0]:.1%}, "
        f"pc2={pca.explained_variance_ratio_[1]:.1%}"
    )


def build_agent_cluster_map(agent_features: pd.DataFrame, k: int) -> dict:
    # agent name: cluster id; consumed by downstream composition analysis
    col = f"cluster_k{k}"
    return dict(zip(agent_features["agents"], agent_features[col].astype(int)))

def print_feature_correlations(agent_features: pd.DataFrame, feat_cols: list, threshold: float = 0.7) -> None:
    corr = agent_features[feat_cols].corr()
    print("feature correlations (clustering inputs):")
    print(corr.round(2).to_string())
    pairs = [
        (feat_cols[i], feat_cols[j], corr.iloc[i, j])
        for i in range(len(feat_cols))
        for j in range(i + 1, len(feat_cols))
        if abs(corr.iloc[i, j]) >= threshold
    ]
    if pairs:
        print(f"\nwarn: |corr| >= {threshold}:")
        for a, b, v in pairs:
            print(f"  {a} <-> {b}: {v:.2f}")

def compute_silhouette_scores(x_scaled: np.ndarray, k_range) -> list:
    return [
        silhouette_score(
            x_scaled, KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(x_scaled).labels_,
        )
        for k in k_range
    ]