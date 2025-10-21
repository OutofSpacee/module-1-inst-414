import pandas as pd

# Import CSV files 
win_rate_by_only_faction = pd.read_csv("collected_data/win_rate_and_points_by_faction.csv")
win_rate_by_detachment = pd.read_csv("collected_data/win_rate_by_detachment.csv")
win_rate_by_faction_and_mission = pd.read_csv("collected_data/win_rate_by_faction_and_mission.csv")
win_rate_by_faction_opponent = pd.read_csv("collected_data/win_rate_by_faction_opponent.csv")

# Clean column headers for all DataFrames
# Each dataset goes through the following . . .
    # - strip leading/trailing whitespace
    # - convert to lowercase
    # - replace spaces with underscores
    # - replace percent signs with the word "percentage"
win_rate_by_only_faction.columns = win_rate_by_only_faction.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%","percentage")
win_rate_by_detachment.columns = win_rate_by_detachment.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%","percentage")
win_rate_by_faction_and_mission.columns = win_rate_by_faction_and_mission.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%","percentage")
win_rate_by_faction_opponent.columns = win_rate_by_faction_opponent.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("%","percentage")

# Fix column names for win_rate_by_faction_opponent:
# The CSV had a misaligned column giving null values    
win_rate_by_faction_opponent = win_rate_by_faction_opponent.drop("opponent", axis=1) # 1) drop the existing "opponent" column (it is redundant/misplaced)
win_rate_by_faction_opponent.rename( # 2) rename the remaining columns so that each field shifts into its correct name
    columns={
        "games": "opponent",
        "wins": "games",
        "losses": "wins",
        "draws": "losses",
        "win_percentage": "draws",
        "col6": "win_percentage"
    },
    inplace=True
)
# mapping key: "games" -> "opponent", "wins" -> "games", "losses" -> "wins",
# "draws" -> "losses", "win_percentage" -> "draws", "col6" -> "win_percentage"

