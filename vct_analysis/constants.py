# role groupings for valorant agents
DUELIST = {"reyna", "jett", "neon", "yoru", "phoenix", "waylay", "raze", "iso"}
SENTINEL = {"cypher", "sage", "vyse", "chamber", "deadlock", "killjoy", "veto"}
INITIATOR = {"breach", "skye", "kayo", "tejo", "sova", "fade", "gekko"}
CONTROLLER = {"brimstone", "viper", "omen", "astra", "clove", "harbor"}

# match identity: full join key across all three source files
KEY = ["tournament", "stage", "match_type", "match_name", "map"]

# renames for each source csv
OVERVIEW_RENAMES = {
    "Tournament": "tournament", "Stage": "stage", "Match Type": "match_type",
    "Match Name": "match_name", "Map": "map", "Player": "player", "Team": "team",
    "Agents": "agents", "Rating": "rating", "Average Combat Score": "acs",
    "Kills": "kills", "Deaths": "deaths", "Assists": "assists",
    "Kills - Deaths (KD)": "kd", "Kill, Assist, Trade, Survive %": "kast",
    "Average Damage Per Round": "adr", "Headshot %": "hs",
    "First Kills": "fk", "First Deaths": "fd",
    "Kills - Deaths (FKD)": "fkd", "Side": "side",
}

KILLS_STATS_RENAMES = {
    "Tournament": "tournament", "Stage": "stage", "Match Type": "match_type",
    "Match Name": "match_name", "Map": "map", "Team": "team", "Player": "player",
    "Agents": "agents_ks", "2k": "k2", "3k": "k3", "4k": "k4", "5k": "k5",
    "1v1": "v1", "1v2": "v2", "1v3": "v3", "1v4": "v4", "1v5": "v5",
    "Econ": "econ", "Spike Plants": "plants", "Spike Defuses": "defuses",
}

MAPS_SCORES_RENAMES = {
    "Tournament": "tournament", "Stage": "stage", "Match Type": "match_type",
    "Match Name": "match_name", "Map": "map",
    "Team A": "team_a", "Team A Score": "a_score",
    "Team B": "team_b", "Team B Score": "b_score",
}

# kills_stats columns are per-map only (no side breakdown)
KILLS_ONLY_COLS = ["k2", "k3", "k4", "k5", "v1", "v2", "v3", "v4", "v5",
                   "econ", "plants", "defuses"]