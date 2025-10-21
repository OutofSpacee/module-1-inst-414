import math  # math utilities
import pandas as pd  # dataframes

# normalizing percentages from 100s to 1s 
# essentially scaling down for ease of use later
def pct_to_float(s: pd.Series) -> pd.Series: 
    """Convert a percentage-like Series (e.g., '51.2%') or numeric to float in [0,1]."""
    if s.dtype.kind in {"f", "i"}:  # numeric dtype
        # Already numeric 0-100 or 0-1; assume 0-100 if > 1
        return s.astype(float) / (100.0 if s.max() > 1 else 1.0)  # normalize to [0,1]
    # Object/strings
    return (
        s.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)  # remove percent sign
        .replace({"nan": None, "": None})  # treat empty/nan strings as None
        .astype(float)
        / 100.0  # convert to fraction
    )

def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)  # clamp away from 0/1 to avoid infinities
    return math.log(p / (1 - p))  # log-odds


def inv_logit(l: float) -> float:
    return 1.0 / (1.0 + math.exp(-l))  # logistic transform -> probability


# Stabilize win-rate estimates for rows with few games (prevents extreme/unstable log-odds).
def smoothed_logit(wins: float, games: float, prior_p: float, k: float = 20.0) -> float:
    """Laplace/empirical Bayes smoothing to stabilize low-sample rows.

    p_hat = (wins + k * prior_p) / (games + k)
    returns logit(p_hat)
    """
    if games is None or pd.isna(games) or games <= 0:  # no data -> use prior
        return logit(prior_p)
    p_hat = (wins + k * prior_p) / (games + k)  # smoothed proportion
    return logit(p_hat)  # return log-odds of smoothed proportion


# Load and clean the CSV tables used by the prediction pipeline.
    # This centralizes parsing, header normalization, type coercion, and a few heuristics for fixing malformed CSV exports 
    # (especially the faction-vs-opponent file). Keeping this logic here keeps the rest of the script focused on modeling and prediction.
def load_tables(base_path: str = "collected_data") -> dict:
    # Faction-only
    df_f = pd.read_csv(f"{base_path}/win_rate_and_points_by_faction.csv")  # load faction table
    # Normalize columns
    df_f.columns = (
        df_f.columns.str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("%", "percentage", regex=False)
    )  # normalize header names
    # Compute wins/games (wins provided), ensure numeric
    df_f["wins"] = pd.to_numeric(df_f["wins"], errors="coerce")  # coerce numeric
    df_f["games"] = pd.to_numeric(df_f["games"], errors="coerce")  # coerce numeric
    df_f["faction"] = df_f["faction"].astype(str)  # ensure string

    # Faction + mission + deployment
    df_fmd = pd.read_csv(f"{base_path}/win_rate_by_faction_and_mission.csv")  # load fmd table
    df_fmd.columns = (
        df_fmd.columns.str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("%", "percentage", regex=False)
    )  # normalize headers
    # Convert types
    df_fmd["games"] = pd.to_numeric(df_fmd["games"], errors="coerce")  # games numeric
    df_fmd["win_percentage"] = pct_to_float(df_fmd["win_percentage"])  # in [0,1]
    df_fmd["wins"] = (df_fmd["games"] * df_fmd["win_percentage"]).round(0)  # reconstruct wins
    for col in ["faction", "mission", "deployment"]:
        df_fmd[col] = df_fmd[col].astype(str)  # ensure strings

    # Faction vs opponent (CSV appears misaligned; fix like in user's cleaner)
    df_fo = pd.read_csv(f"{base_path}/win_rate_by_faction_opponent.csv")  # load fo table
    df_fo.columns = (
        df_fo.columns.str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("%", "percentage", regex=False)
    )  # normalize headers
    # If first column is a broken/empty 'opponent' column, drop it and shift names as in user's code
    if "opponent" in df_fo.columns:
        try:
            # Check if the first column is mostly empty strings
            if df_fo["opponent"].isna().mean() > 0.5 or (df_fo["opponent"].astype(str) == "").mean() > 0.5:
                df_fo = df_fo.drop("opponent", axis=1)  # drop malformed column
                df_fo.rename(
                    columns={
                        "games": "opponent",
                        "wins": "games",
                        "losses": "wins",
                        "draws": "losses",
                        "win_percentage": "draws",
                        "col6": "win_percentage",
                    },
                    inplace=True,
                )  # shift columns to expected names
        except Exception:
            pass  # ignore errors in heuristic fix
    # Ensure types and names
    # After fix, expected columns: opponent (name), games, wins, losses, draws, win_percentage, faction
    # Some rows can still be malformed; coerce and drop bad
    if "win_percentage" in df_fo.columns:
        df_fo["win_percentage"] = pct_to_float(df_fo["win_percentage"])  # [0,1]
    # Coerce numerics
    for c in ["games", "wins", "losses", "draws"]:
        if c in df_fo.columns:
            df_fo[c] = pd.to_numeric(df_fo[c], errors="coerce")  # numeric coercion
    # Standardize strings
    for c in ["faction", "opponent"]:
        if c in df_fo.columns:
            df_fo[c] = df_fo[c].astype(str)  # ensure strings
    # Drop rows missing essentials
    df_fo = df_fo.dropna(subset=["faction", "opponent", "games"])  # remove bad rows

    return {"faction": df_f, "fmd": df_fmd, "fo": df_fo}  # return dict of tables

# Constructs smoothed log-odds "effects" (global prior, faction baselines, faction+mission+deployment, faction vs opponent)
# This was included to centralize and standardize how win-rate information is converted into regularized log-odds inputs for downstream prediction/analysis.
def build_effects(tables: dict, k: float = 20.0):
    df_f = tables["faction"].copy()  # faction baseline table
    df_fmd = tables["fmd"].copy()  # faction-mission-deployment table
    df_fo = tables["fo"].copy()  # faction-opponent table

    # Global prior from faction-only table
    total_wins = df_f["wins"].sum()  # total wins
    total_games = df_f["games"].sum()  # total games
    prior_p = (total_wins / total_games) if total_games > 0 else 0.5  # global prior prob
    l0 = logit(prior_p)  # global log-odds

    # Faction baselines
    fx = (
        df_f[["faction", "wins", "games"]]
        .groupby("faction", as_index=False)
        .sum()
        .assign(l_f=lambda d: d.apply(lambda r: smoothed_logit(r.wins, r.games, prior_p, k), axis=1))
    )  # compute smoothed faction log-odds

    # Faction + mission + deployment (wins from table)
    fmd = df_fmd[["faction", "mission", "deployment", "wins", "games"]].copy()  # select cols
    fmd["l_fmd"] = fmd.apply(lambda r: smoothed_logit(r.wins, r.games, prior_p, k), axis=1)  # smoothed l for fmd

    # Faction vs opponent (wins from table)
    # If wins missing but win_percentage present, reconstruct wins
    if fmd["wins"].isna().any():
        fmd["wins"] = (fmd["games"] * pct_to_float(fmd["win_percentage"]) if "win_percentage" in fmd.columns else fmd["wins"])  # type: ignore  # attempt reconstruct wins

    fo = df_fo[["faction", "opponent", "wins", "games"]].copy()  # select cols
    # If 'wins' column is empty due to malformed CSV, but win_percentage exists, compute wins
    if fo["wins"].isna().mean() > 0.5 and "win_percentage" in df_fo.columns:
        fo["wins"] = (df_fo["games"] * df_fo["win_percentage"]).round(0)  # reconstruct wins
    fo["l_fo"] = fo.apply(lambda r: smoothed_logit(r.wins, r.games, prior_p, k), axis=1)  # smoothed l for fo

    return l0, prior_p, fx, fmd, fo  # return computed effects

# Combines baseline log-odds and two faction-specific effect tables into predicted win probabilities for every valid mission/deployment/opponent combination. 
# Included to centralize prediction logic and enforce minimum-sample filtering.
def combine_predictions(l0, fx, fmd, fo, min_games_fmd: int = 30, min_games_fo: int = 30):
    # Merge faction baselines into the tables
    fmd = fmd.merge(fx[["faction", "l_f"]], on="faction", how="left")  # attach l_f to fmd
    fo = fo.merge(fx[["faction", "l_f"]], on="faction", how="left")  # attach l_f to fo

    # Filter by minimum games to avoid tiny samples
    fmd_filt = fmd[fmd["games"] >= min_games_fmd].copy()  # keep larger fmd rows
    fo_filt = fo[fo["games"] >= min_games_fo].copy()  # keep larger fo rows

    # For each faction, cross join fmd rows with fo rows
    results = []
    for faction, fmd_g in fmd_filt.groupby("faction"):  # iterate factions
        fo_g = fo_filt[fo_filt["faction"] == faction]  # matching fo rows
        if fo_g.empty or fmd_g.empty:
            continue  # skip if no combos
        # Cartesian join via product
        for _, r1 in fmd_g.iterrows():
            for _, r2 in fo_g.iterrows():
                l_comb = l0 + r1["l_fmd"] + r2["l_fo"] - r1["l_f"]  # combine effects (avoid double-counting l_f)
                p_hat = inv_logit(l_comb)  # predicted probability
                results.append(
                    {
                        "faction": faction,
                        "mission": r1["mission"],
                        "deployment": r1["deployment"],
                        "opponent": r2["opponent"],
                        "predicted_win_prob": p_hat,
                        "games_fmd": int(r1["games"]),
                        "games_fo": int(r2["games"]),
                    }
                )  # append prediction row

    pred_df = pd.DataFrame(results)  # build dataframe
    if pred_df.empty:
        return pred_df  # return empty df if no results
    pred_df.sort_values(["predicted_win_prob", "games_fmd", "games_fo"], ascending=[False, False, False], inplace=True)  # rank results
    return pred_df  # return ranked predictions


def predict_best_with_opponent(l0, fx, fmd, fo, min_games_fmd: int = 30, min_games_fo: int = 30):
    """Wrapper that returns ranked predictions including opponent.

    This simply calls combine_predictions but provides a clear API name for
    'best setup including opponent' use-cases.
    """
    # Delegate to combine_predictions to generate ranked predictions that include opponent-specific effects.
    return combine_predictions(l0, fx, fmd, fo, min_games_fmd=min_games_fmd, min_games_fo=min_games_fo)

# Predict the best faction/mission/deployment setups ignoring the opponent.
def predict_best_general(fmd, fx, l0, min_games_fmd: int = 30):
    """Predict best (faction, mission, deployment) without considering opponent.
    
    Uses faction baseline + mission/deployment deviation.
    """
    fmd_filt = fmd[fmd["games"] >= min_games_fmd].copy()  # filter by games
    fmd_filt = fmd_filt.merge(fx[["faction", "l_f"]], on="faction", how="left")  # attach l_f
    
    # Combined log-odds: l0 + (l_fmd - l_f) + l_f = l0 + l_fmd
    # Simplifies to just using l_fmd directly since l_f cancels
    fmd_filt["predicted_win_prob"] = fmd_filt["l_fmd"].apply(inv_logit)  # convert to prob
    
    results = fmd_filt[["faction", "mission", "deployment", "predicted_win_prob", "games"]].copy()  # select cols
    results.rename(columns={"games": "games_fmd"}, inplace=True)  # rename games
    results.sort_values("predicted_win_prob", ascending=False, inplace=True)  # sort descending
    return results  # return general predictions

# Predict overall faction win probability averaged across missions and deployments.
def predict_faction_overall(fmd, fx, l0, min_games_fmd: int = 30):
    """Predict overall faction win probability averaged across all missions and deployments.
    
    This simulates tournament play where mission and deployment are random.
    Uses weighted average by number of games to account for sample size.
    """
    fmd_filt = fmd[fmd["games"] >= min_games_fmd].copy()  # filter scenarios
    fmd_filt = fmd_filt.merge(fx[["faction", "l_f"]], on="faction", how="left")  # attach l_f
    fmd_filt["predicted_win_prob"] = fmd_filt["l_fmd"].apply(inv_logit)  # convert to prob
    
    # Weighted average: sum(prob * games) / sum(games) per faction
    faction_agg = (
        fmd_filt.groupby("faction")
        .apply(
            lambda g: pd.Series({
                "avg_win_prob": (g["predicted_win_prob"] * g["games"]).sum() / g["games"].sum(),  # weighted avg
                "total_games": g["games"].sum(),
                "num_scenarios": len(g),
            })
        )
        .reset_index()
    )  # aggregated faction stats
    
    faction_agg.sort_values("avg_win_prob", ascending=False, inplace=True)  # sort by avg prob
    return faction_agg  # return faction-level rankings

# calls all the functions and prints it all dolled up
def main():
    tables = load_tables()  # load CSVs into tables
    l0, prior_p, fx, fmd, fo = build_effects(tables, k=20.0)  # compute effects
    
    # 1. Predict best setups considering opponent
    pred_df = combine_predictions(l0, fx, fmd, fo, min_games_fmd=30, min_games_fo=30)  # combine effects

    if pred_df.empty:
        print("No predictions generated. Check input data formatting.")  # notify user
        return

    # Save full predictions and print top 20
    out_path = "predicted_best_setups.csv"  # output path
    pred_df.to_csv(out_path, index=False)  # save CSV
    print(f"Wrote ranked predictions to {out_path} (rows={len(pred_df)})")  # report
    print()
    print("Top 20 setups (with opponent):")
    print(
        pred_df.head(20)[
            [
                "faction",
                "mission",
                "deployment",
                "opponent",
                "predicted_win_prob",
                "games_fmd",
                "games_fo",
            ]
        ].to_string(index=False, formatters={"predicted_win_prob": lambda x: f"{x:.3f}"})  # pretty print top rows
    )
    
    # 2. Predict best general setups (no opponent)
    print("\n" + "="*80 + "\n")
    general_df = predict_best_general(fmd, fx, l0, min_games_fmd=30)  # general predictions
    out_path_general = "predicted_best_general_setups.csv"  # output path
    general_df.to_csv(out_path_general, index=False)  # save CSV
    print(f"Wrote general predictions to {out_path_general} (rows={len(general_df)})")  # report
    print()
    print("Top 20 general setups (no opponent):")
    print(
        general_df.head(20)[
            ["faction", "mission", "deployment", "predicted_win_prob", "games_fmd"]
        ].to_string(index=False, formatters={"predicted_win_prob": lambda x: f"{x:.3f}"})  # pretty print
    )
    
    # 3. Predict overall faction performance (averaged across missions/deployments)
    print("\n" + "="*80 + "\n")
    faction_overall = predict_faction_overall(fmd, fx, l0, min_games_fmd=30)  # faction overall ranking
    out_path_overall = "predicted_faction_overall.csv"  # output path
    faction_overall.to_csv(out_path_overall, index=False)  # save CSV
    print(f"Wrote faction overall rankings to {out_path_overall} (rows={len(faction_overall)})")  # report
    print()
    print("Faction overall win probability (tournament average):")
    print(
        faction_overall.to_string(
            index=False, 
            formatters={"avg_win_prob": lambda x: f"{x:.3f}"}
        )  # pretty print faction table
    )


if __name__ == "__main__":
    main()  # run when executed as script
