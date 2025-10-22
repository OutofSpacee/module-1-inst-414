import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Format a probability (0-1) as a percentage string like "12.3%".
def fmt_prob(x: float) -> str:
    return f"{x*100:.1f}%"


# Format win rate for display
def fmt_winrate(x: float) -> str:
    return f"{x:.1f}%"


# Load predictions CSV and validate required columns.
def load_predictions(csv_path: str = "predicted_best_setups.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure expected columns exist
    needed = {"faction", "mission", "deployment", "opponent", "predicted_win_prob"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in predictions CSV: {missing}")
    return df


# Plot the top 20 opponent-specific setups as a horizontal bar chart.
def plot_top20_bar(df: pd.DataFrame, out_dir: Path) -> None:
    top = df.sort_values("predicted_win_prob", ascending=False).head(20).copy()
    top["label"] = (
        top["faction"]
        + " | "
        + top["mission"]
        + " | "
        + top["deployment"]
        + " | vs "
        + top["opponent"]
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(
        data=top,
        y="label",
        x="predicted_win_prob",
        hue="faction",
        dodge=False,
        ax=ax,
        palette="viridis",
    )
    ax.legend(loc="lower right", fontsize=8)

    ax.set_title("Top 20 predicted setups by win probability")
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("")
    # Add percentage labels
    for i, v in enumerate(top["predicted_win_prob" ]):
        ax.text(v + 0.005, i, fmt_prob(v), va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "top20_setups.png", dpi=150)
    plt.close(fig)


# For each faction, create a mission x deployment heatmap using the max win prob across opponents.
def plot_faction_heatmaps(df: pd.DataFrame, out_dir: Path, max_factions: int = 5) -> None:
    factions = (
        df.groupby("faction")
        .size()
        .sort_values(ascending=False)
        .head(max_factions)
        .index.tolist()
    )
    for faction in factions:
        sub = df[df["faction"] == faction].copy()
        agg = (
            sub.groupby(["mission", "deployment"], as_index=False)["predicted_win_prob"].max()
        )
        pivot = agg.pivot(index="mission", columns="deployment", values="predicted_win_prob")

        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(pivot))))
        sns.heatmap(
            pivot,
            cmap="RdYlGn",
            vmin=0.35,
            vmax=0.65,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Predicted win prob"},
            ax=ax,
        )
        ax.set_title(f"{faction}: best mission x deployment (max vs opponent)")
        ax.set_xlabel("Deployment")
        ax.set_ylabel("Mission")
        plt.tight_layout()
        fig.savefig(out_dir / f"heatmap_{faction.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)


# Plot faction overall rankings with top 3 highlighted
def plot_faction_overall(csv_path: Path, out_dir: Path) -> None:
    """Plot faction overall rankings with top performers highlighted."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("avg_win_prob", ascending=True)  # ascending for horizontal bars
    
    # Highlight top 3 and bottom 5
    colors = []
    for faction in df["faction"]:
        if faction in ["Imperial Knights", "Chaos Knights", "Death Guard"]:
            colors.append("#2ecc71")  # green for top
        elif faction in ["Space Marines", "Chaos Space Marines", "Astra Militarum", "Orks", "T'au"]:
            colors.append("#e74c3c")  # red for bottom
        else:
            colors.append("#95a5a6")  # gray for middle
    
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(df["faction"], df["avg_win_prob"], color=colors, alpha=0.8)
    
    ax.set_title("Faction Overall Win Probability\n(Imperial Knights, Chaos Knights, Death Guard lead)", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Win Probability", fontsize=12)
    ax.set_ylabel("")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.3, linewidth=1.5, label="50% baseline")
    
    # Add percentage labels
    for i, (idx, row) in enumerate(df.iterrows()):
        color = "white" if row["avg_win_prob"] > 0.52 else "black"
        ax.text(
            row["avg_win_prob"] - 0.008 if row["avg_win_prob"] > 0.52 else row["avg_win_prob"] + 0.008,
            i,
            fmt_prob(row["avg_win_prob"]),
            va="center",
            ha="right" if row["avg_win_prob"] > 0.52 else "left",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Top 3 (Knights & Death Guard)'),
        Patch(facecolor='#95a5a6', label='Mid-tier'),
        Patch(facecolor='#e74c3c', label='Bottom 5')
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(out_dir / "faction_overall_rankings.png", dpi=150)
    plt.close(fig)


# Plot top factions performance across missions
def plot_top_factions_by_mission(raw_data_path: Path, out_dir: Path) -> None:
    """Show how top 3 factions perform across different missions."""
    df = pd.read_csv(raw_data_path / "collected_data" / "win_rate_by_faction_and_mission.csv")
    
    # Filter to top 3 factions
    top_factions = ["Imperial Knights", "Chaos Knights", "Death Guard"]
    df_top = df[df["Faction"].isin(top_factions)].copy()
    
    # Parse win percentage
    df_top["Win %"] = df_top["Win %"].str.replace("%", "").astype(float)
    
    # Aggregate by faction and mission
    mission_perf = df_top.groupby(["Faction", "Mission"]).agg({
        "Win %": "mean",
        "Games": "sum"
    }).reset_index()
    
    # Filter missions with at least 100 games
    mission_perf = mission_perf[mission_perf["Games"] >= 100]
    
    # Pivot for heatmap
    pivot = mission_perf.pivot(index="Mission", columns="Faction", values="Win %")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        vmin=45,
        vmax=65,
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Win Rate (%)"},
        ax=ax,
        center=50,
    )
    ax.set_title("Top 3 Factions: Win Rate by Mission Type\n(Knights and Death Guard consistently above 50%)", 
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Faction", fontsize=11)
    ax.set_ylabel("Mission", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / "top_factions_by_mission.png", dpi=150)
    plt.close(fig)


# Plot deployment performance for top factions
def plot_deployment_performance(raw_data_path: Path, out_dir: Path) -> None:
    """Show how top factions perform on different deployments."""
    df = pd.read_csv(raw_data_path / "collected_data" / "win_rate_by_faction_and_mission.csv")
    
    top_factions = ["Imperial Knights", "Chaos Knights", "Death Guard"]
    df_top = df[df["Faction"].isin(top_factions)].copy()
    
    df_top["Win %"] = df_top["Win %"].str.replace("%", "").astype(float)
    
    # Aggregate by faction and deployment
    deploy_perf = df_top.groupby(["Faction", "Deployment"]).agg({
        "Win %": "mean",
        "Games": "sum"
    }).reset_index()
    
    # Filter deployments with at least 50 games
    deploy_perf = deploy_perf[deploy_perf["Games"] >= 50]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=deploy_perf,
        x="Deployment",
        y="Win %",
        hue="Faction",
        ax=ax,
        palette=["#3498db", "#9b59b6", "#16a085"],
    )
    
    ax.set_title("Top Factions: Win Rate by Deployment\n(Consistent performance across all map layouts)", 
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Deployment Type", fontsize=11)
    ax.set_ylabel("Average Win Rate (%)", fontsize=11)
    ax.axhline(50, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="50% baseline")
    ax.legend(loc="lower right", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "deployment_performance.png", dpi=150)
    plt.close(fig)


# Compare detachment performance for top factions
def plot_top_detachments(raw_data_path: Path, out_dir: Path) -> None:
    """Show the strongest detachments by faction."""
    df = pd.read_csv(raw_data_path / "collected_data" / "win_rate_by_detachment.csv")
    
    df["Win %"] = df["Win %"].str.replace("%", "").astype(float)
    
    # Filter to detachments with at least 1000 games and win rate > 52%
    df_strong = df[(df["Games"] >= 1000) & (df["Win %"] >= 52)].copy()
    df_strong = df_strong.sort_values("Win %", ascending=False).head(15)
    
    # Extract faction from detachment name
    df_strong["Faction"] = df_strong["Detachment"].str.split(" - ").str[0]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = sns.barplot(
        data=df_strong,
        y="Detachment",
        x="Win %",
        hue="Faction",
        dodge=False,
        ax=ax,
        palette="viridis",
    )
    
    ax.set_title("Strongest Detachments (1000+ games, 52%+ win rate)\nNoble Lance, Infernal Lance, and Death Guard variants dominate", 
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Win Rate (%)", fontsize=11)
    ax.set_ylabel("")
    ax.axvline(50, color="red", linestyle="--", alpha=0.3, linewidth=1.5)
    
    # Add win rate labels
    for i, v in enumerate(df_strong["Win %"]):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    fig.savefig(out_dir / "top_detachments.png", dpi=150)
    plt.close(fig)


# Plot top 3 factions across all mission/deployment combinations
def plot_top3_mission_deployment_grid(raw_data_path: Path, out_dir: Path) -> None:
    """Create a comprehensive view of top 3 factions across mission/deployment combos."""
    df = pd.read_csv(raw_data_path / "collected_data" / "win_rate_by_faction_and_mission.csv")
    
    top_factions = ["Imperial Knights", "Chaos Knights", "Death Guard"]
    df_top = df[df["Faction"].isin(top_factions)].copy()
    
    # Parse win percentage
    df_top["Win %"] = df_top["Win %"].str.replace("%", "").astype(float)
    
    # Create separate heatmaps for each faction
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for idx, faction in enumerate(top_factions):
        df_faction = df_top[df_top["Faction"] == faction].copy()
        
        # Aggregate by mission and deployment
        agg = df_faction.groupby(["Mission", "Deployment"]).agg({
            "Win %": "mean",
            "Games": "sum"
        }).reset_index()
        
        # Filter to combos with at least 20 games
        agg = agg[agg["Games"] >= 20]
        
        # Pivot for heatmap
        pivot = agg.pivot(index="Mission", columns="Deployment", values="Win %")
        
        # Create heatmap
        sns.heatmap(
            pivot,
            cmap="RdYlGn",
            vmin=40,
            vmax=70,
            annot=True,
            fmt=".1f",
            cbar_kws={"label": "Win Rate (%)"},
            ax=axes[idx],
            center=50,
            linewidths=0.5,
            linecolor='gray',
        )
        
        axes[idx].set_title(f"{faction}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Deployment", fontsize=10)
        axes[idx].set_ylabel("Mission" if idx == 0 else "", fontsize=10)
        
        # Rotate labels for readability
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right", fontsize=8)
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0, fontsize=8)
    
    fig.suptitle("Top 3 Factions: Win Rate by Mission × Deployment\n(Knights and Death Guard consistently above 50%)", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(out_dir / "top3_mission_deployment_grid.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


# Show Imperial Knights matchup matrix
def plot_knights_matchups(raw_data_path: Path, out_dir: Path) -> None:
    """Show Imperial Knights win rates against all factions."""
    df = pd.read_csv(raw_data_path / "collected_data" / "win_rate_by_faction_opponent.csv")
    
    # Handle the malformed CSV structure - the first column is often empty, opponent name is in "Games" column
    # Original columns: Opponent,Games,Wins,Losses,Draws,Win %,col6,Faction
    # Where first "Opponent" column is empty and actual opponent is in what's labeled "Games"
    if "Opponent" in df.columns and df["Opponent"].isna().mean() > 0.5:
        # Drop the empty first column and rename
        df = df.drop(columns=["Opponent"])
        df = df.rename(columns={
            "Games": "Opponent",
            "Wins": "Games", 
            "Losses": "Wins",
            "Draws": "Losses",
            "Win %": "Draws",
            "col6": "Win %"
        })
    
    # Filter to Imperial Knights rows
    df_knights = df[df["Faction"] == "Imperial Knights"].copy()
    
    # Clean win percentage - handle both string and numeric formats
    if df_knights["Win %"].dtype == object:
        df_knights["Win %"] = df_knights["Win %"].str.replace("%", "").astype(float)
    else:
        # Already numeric, just ensure it's float
        df_knights["Win %"] = df_knights["Win %"].astype(float)
    
    df_knights = df_knights.sort_values("Win %", ascending=True)
    
    # Color code: green for >55%, red for <45%, gray otherwise
    colors = []
    for wr in df_knights["Win %"]:
        if wr >= 55:
            colors.append("#2ecc71")
        elif wr <= 45:
            colors.append("#e74c3c")
        else:
            colors.append("#95a5a6")
    
    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(df_knights["Opponent"], df_knights["Win %"], color=colors, alpha=0.8)
    
    ax.set_title("Imperial Knights: Win Rate by Opponent\nStrong against most, but countered by Aeldari & GSC", 
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Win Rate vs Opponent (%)", fontsize=11)
    ax.set_ylabel("")
    ax.axvline(50, color="black", linestyle="--", alpha=0.3, linewidth=1.5)
    
    # Add win rate labels
    for i, (idx, row) in enumerate(df_knights.iterrows()):
        color = "white" if row["Win %"] > 55 else "black"
        ax.text(
            row["Win %"] - 1 if row["Win %"] > 55 else row["Win %"] + 1,
            i,
            f"{row['Win %']:.1f}%",
            va="center",
            ha="right" if row["Win %"] > 55 else "left",
            fontsize=8,
            fontweight="bold" if row["Win %"] >= 55 or row["Win %"] <= 45 else "normal",
            color=color,
        )
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Strong matchup (≥55%)'),
        Patch(facecolor='#95a5a6', label='Even matchup (45-55%)'),
        Patch(facecolor='#e74c3c', label='Weak matchup (≤45%)')
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(out_dir / "imperial_knights_matchups.png", dpi=150)
    plt.close(fig)


# Plot the top 20 general setups (no opponent) as a horizontal bar chart.
def plot_general_setups_top20(csv_path: Path, out_dir: Path) -> None:
    """Plot top 20 general setups (no opponent) as a horizontal bar chart."""
    df = pd.read_csv(csv_path)
    top = df.head(20).copy()
    top["label"] = top["faction"] + " | " + top["mission"] + " | " + top["deployment"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=top,
        y="label",
        x="predicted_win_prob",
        hue="faction",
        dodge=False,
        ax=ax,
        palette="viridis",
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    
    ax.set_title("Top 20 General Setups (No Opponent)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Win Probability")
    ax.set_ylabel("")
    
    # Add percentage labels
    for i, v in enumerate(top["predicted_win_prob"]):
        ax.text(v + 0.005, i, fmt_prob(v), va="center", fontsize=8)
    
    plt.tight_layout()
    fig.savefig(out_dir / "top20_general_setups.png", dpi=150)
    plt.close(fig)


# Main routine: load CSVs, generate figures, and save them to the figures/ directory.
def main():
    root = Path(__file__).parent
    pred_csv = root / "prediction_data" / "predicted_best_setups.csv"
    general_csv = root / "prediction_data" / "predicted_best_general_setups.csv"
    overall_csv = root / "prediction_data" / "predicted_faction_overall.csv"
    
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations for data analysis...")
    print()
    
    # 1. Faction overall rankings (KEY FINDING: Knights & Death Guard on top)
    if overall_csv.exists():
        print("  [1/8] Creating faction overall rankings...")
        plot_faction_overall(overall_csv, out_dir)
    
    # 2. Top detachments (SUPPORTS: Noble Lance, Infernal Lance dominance)
    print("  [2/8] Creating top detachments chart...")
    plot_top_detachments(root, out_dir)
    
    # 3. Top factions by mission (SHOWS: consistency across missions)
    print("  [3/8] Creating mission performance comparison...")
    plot_top_factions_by_mission(root, out_dir)
    
    # 4. Deployment performance (SHOWS: Knights strong on all deployments)
    print("  [4/8] Creating deployment performance comparison...")
    plot_deployment_performance(root, out_dir)
    
    # 5. Mission × Deployment grid for top 3 (COMPREHENSIVE VIEW)
    print("  [5/8] Creating mission × deployment grid for top 3...")
    plot_top3_mission_deployment_grid(root, out_dir)
    
    # 6. Imperial Knights matchups (REVEALS: Aeldari/GSC counters)
    print("  [6/8] Creating Imperial Knights matchup matrix...")
    plot_knights_matchups(root, out_dir)
    
    # 7. General predictions (no opponent)
    if general_csv.exists():
        print("  [7/8] Creating top general setups chart...")
        plot_general_setups_top20(general_csv, out_dir)
    
    # 8. Opponent-specific predictions and heatmaps
    if pred_csv.exists():
        print("  [8/8] Creating detailed faction heatmaps...")
        df = load_predictions(str(pred_csv))
        plot_faction_heatmaps(df, out_dir, max_factions=3)  # Only top 3
    
    print()
    print(f"✓ Saved {8} figures to {out_dir}/")
    print()
    print("Key visualizations supporting Data Analysis section:")
    print("  → faction_overall_rankings.png: Shows Knights & Death Guard dominance")
    print("  → top_detachments.png: Noble Lance, Infernal Lance, Virulent Vectorium")
    print("  → top_factions_by_mission.png: Consistency across mission types")
    print("  → deployment_performance.png: Strong performance on all deployments")
    print("  → top3_mission_deployment_grid.png: Complete mission × deployment view")
    print("  → imperial_knights_matchups.png: Countered by Aeldari/GSC")
    print("  → top_general_setups.png: Best mission/deployment combinations")
    print("  → heatmap_*.png: Per-faction mission × deployment analysis")


if __name__ == "__main__":
    main()
