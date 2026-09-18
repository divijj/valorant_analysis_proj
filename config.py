# central config: paths, thresholds, seed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / ".." / "vct_2026"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# raw csv paths
OVERVIEW_CSV = DATA_DIR / "matches" / "overview.csv"
KILLS_STATS_CSV = DATA_DIR / "matches" / "kills_stats.csv"
MAPS_SCORES_CSV = DATA_DIR / "matches" / "maps_scores.csv"
TEAMS_AGENTS_CSV = DATA_DIR / "agents" / "teams_picked_agents.csv"
WIN_LOSS_CSV = DATA_DIR / "matches" / "win_loss_methods_count.csv"

# derived outputs
MASTER_CSV = OUTPUTS_DIR / "master.csv"
AGENT_FEATURES_CSV = OUTPUTS_DIR / "agent_features.csv"

# thresholds

# drop agents with too few player-maps
MIN_MAPS_PER_AGENT = 5         

# skip regression for clusters too small 
MIN_REGRESSION_SAMPLES = 100     

# cell sample-size floor for heatmaps
MIN_HEATMAP_SAMPLES = 5          

# clustering
# run both
K_VALUES = [4, 6]

# which k feeds composition + regression
K_DOWNSTREAM = 4

RANDOM_SEED = 42

OUTPUTS_DIR.mkdir(exist_ok=True)