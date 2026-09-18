# per-cluster logistic regression with team fixed effects
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import MIN_REGRESSION_SAMPLES

# predictors deliberately exclude rating/acs/kd (composites of others) and econ (~0.89 corr with adr).
# previously caused negative acs coefficients due to collinearity.
DEFAULT_PREDICTORS = ["adr", "kast", "fkd", "clutch_score"]

def build_regression_df(master: pd.DataFrame, agent_features: pd.DataFrame, k: int) -> pd.DataFrame:
    # one row per (player, map, side='both') with cluster id attached.
    # plants.notna() filter drops china kickoff (no kills_stats data there).
    col = f"cluster_k{k}"
    reg = master[
        (master["side"] == "both")
        & (master["role"].notna())
        & (master["plants"].notna())
    ].copy()
    reg = reg.merge(agent_features[["agents", col]], on="agents", how="left")
    reg = reg.dropna(subset=[col])
    reg[col] = reg[col].astype(int)
    reg = reg.rename(columns={col: "cluster"})

    # composite predictors from kills_stats columns
    reg["multikill_score"] = reg["k3"] + 2 * reg["k4"] + 3 * reg["k5"]
    reg["clutch_score"] = reg["v1"] + reg["v2"] + reg["v3"] + reg["v4"] + reg["v5"]
    return reg


def run_per_cluster_regression(
    reg: pd.DataFrame,
    predictors: list = None,
    min_n: int = MIN_REGRESSION_SAMPLES,
) -> pd.DataFrame:
    # for each cluster, fit logistic regression with standardized predictors + team dummies
    # team dummies absorb team strength, so predictor coefs reflect within-team effects
    if predictors is None:
        predictors = DEFAULT_PREDICTORS
    reg = reg.dropna(subset=predictors).copy()
    print(f"regression rows: {len(reg)}")
    print("\ncluster distribution:")
    print(reg.groupby("cluster").size().to_string())
    print("\npredictor correlations (verify <0.7 for non-redundancy):")
    print(reg[predictors].corr().round(2).to_string())

    results = {}
    for c in sorted(reg["cluster"].unique()):
        sub = reg[reg["cluster"] == c].copy()
        if len(sub) < min_n:
            print(f"\n--- skipping cluster_{c} (n={len(sub)} < {min_n}) ---")
            continue

        x_pred = StandardScaler().fit_transform(sub[predictors])
        x_pred_df = pd.DataFrame(x_pred, columns=predictors, index=sub.index)
        x_team = pd.get_dummies(sub["team"], prefix="team", drop_first=True).astype(float)
        x = pd.concat([x_pred_df, x_team], axis=1)
        y = sub["won"].astype(int).values

        model = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        model.fit(x, y)
        # only retain predictor coefs (not the team-dummy ones)
        coefs = pd.Series(model.coef_[0][: len(predictors)], index=predictors)
        results[c] = coefs

        print(f"\n--- cluster_{c} (n={len(sub)}) ---")
        print(coefs.sort_values(key=abs, ascending=False).round(3).to_string())

    coef_table = pd.DataFrame(results)
    coef_table.columns = [f"cluster_{c}" for c in coef_table.columns]
    print("\n=== standardized log-odds coefficients ===")
    print(coef_table.round(3).to_string())
    return coef_table
    