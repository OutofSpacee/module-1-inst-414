import os
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Format a probability (0-1) as a percentage string like "12.3%".
def fmt_prob(x: float) -> str:
    return f"{x*100:.1f}%"


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


# Plot faction overall rankings as a horizontal bar chart showing average win probabilities.
def plot_faction_overall(csv_path: Path, out_dir: Path) -> None:
    """Plot faction overall rankings as a horizontal bar chart."""
    df = pd.read_csv(csv_path)
    df = df.sort_values("avg_win_prob", ascending=True)  # ascending for horizontal bars
    
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(df))))
    sns.barplot(
        data=df,
        y="faction",
        x="avg_win_prob",
        hue="faction",
        dodge=False,
        ax=ax,
        palette="viridis",
        legend=False,
    )
    
    ax.set_title("Faction Overall Win Probability (Tournament Average)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Win Probability")
    ax.set_ylabel("")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1)
    
    # Add percentage labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(
            row["avg_win_prob"] + 0.005,
            i,
            fmt_prob(row["avg_win_prob"]),
            va="center",
            fontsize=9,
        )
    
    plt.tight_layout()
    fig.savefig(out_dir / "faction_overall_rankings.png", dpi=150)
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
    pred_csv = root / "predicted_best_setups.csv"
    general_csv = root / "predicted_best_general_setups.csv"
    overall_csv = root / "predicted_faction_overall.csv"
    
    if not pred_csv.exists():
        raise SystemExit(
            "predicted_best_setups.csv not found. Run predict_best_setup.py first to generate predictions."
        )

    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Opponent-specific predictions
    df = load_predictions(str(pred_csv))
    plot_top20_bar(df, out_dir)
    plot_faction_heatmaps(df, out_dir, max_factions=5)
    
    # 2. General predictions (no opponent)
    if general_csv.exists():
        plot_general_setups_top20(general_csv, out_dir)
    
    # 3. Faction overall rankings
    if overall_csv.exists():
        plot_faction_overall(overall_csv, out_dir)
    
    print(f"Saved figures to {out_dir}:")
    print("  - top20_setups.png (opponent-specific)")
    print("  - top20_general_setups.png (no opponent)")
    print("  - faction_overall_rankings.png (tournament average)")
    print("  - heatmap_<Faction>.png (per-faction mission x deployment)")


if __name__ == "__main__":
    main()
