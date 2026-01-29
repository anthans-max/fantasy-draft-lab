import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.scoring import calculate_fantasy_points


st.set_page_config(page_title="Fantasy Draft Analytics Demo", layout="wide")

DATA_PATH = REPO_ROOT / "data" / "sample_players.csv"
LEAGUE_TEAMS = 12


@st.cache_data
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["draft_round"] = df["adp_overall"].apply(lambda adp: math.ceil(adp / LEAGUE_TEAMS))
    return df


def add_scoring(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fantasy_points_ppr"] = df.apply(
        lambda row: calculate_fantasy_points(
            rushing_yards=row["rushing_yards"],
            receiving_yards=row["receiving_yards"],
            receptions=row["receptions"],
            touchdowns=row["touchdowns"],
            scoring_format="ppr",
        ),
        axis=1,
    )
    df["fantasy_points_half_ppr"] = df.apply(
        lambda row: calculate_fantasy_points(
            rushing_yards=row["rushing_yards"],
            receiving_yards=row["receiving_yards"],
            receptions=row["receptions"],
            touchdowns=row["touchdowns"],
            scoring_format="half_ppr",
        ),
        axis=1,
    )
    df["fantasy_points_per_game_ppr"] = df["fantasy_points_ppr"] / df["games_played"]
    df["fantasy_points_per_game_half_ppr"] = df["fantasy_points_half_ppr"] / df["games_played"]
    return df


def plot_value_by_round(df: pd.DataFrame, ppg_column: str):
    summary = (
        df.groupby(["draft_round", "position"], as_index=False)[ppg_column]
        .mean()
        .rename(columns={ppg_column: "avg_fppg"})
    )
    fig = px.bar(
        summary,
        x="draft_round",
        y="avg_fppg",
        color="position",
        barmode="group",
        title="Value by Draft Round (Avg FPPG)",
        labels={"draft_round": "Draft Round", "avg_fppg": "Avg Fantasy Points/Game"},
    )
    fig.update_layout(xaxis=dict(dtick=1))
    return fig


def plot_adp_vs_fppg(df: pd.DataFrame, ppg_column: str):
    fig = px.scatter(
        df,
        x="adp_overall",
        y=ppg_column,
        color="position",
        hover_name="player",
        title="ADP vs Fantasy Points Per Game",
        labels={"adp_overall": "ADP (Overall)", ppg_column: "Fantasy Points/Game"},
    )
    fig.update_layout(xaxis=dict(autorange="reversed"))
    return fig


def main():
    st.title("Fantasy Football Draft Analytics")
    st.caption("Demo app for draft value exploration (local CSV data).")

    df = add_scoring(load_players())
    scoring_format = st.selectbox("Scoring format", ["ppr", "half_ppr"], index=0)
    points_column = (
        "fantasy_points_ppr" if scoring_format == "ppr" else "fantasy_points_half_ppr"
    )
    ppg_column = (
        "fantasy_points_per_game_ppr"
        if scoring_format == "ppr"
        else "fantasy_points_per_game_half_ppr"
    )
    df_view = df.copy()
    df_view["fantasy_points"] = df_view[points_column]
    df_view["fantasy_points_per_game"] = df_view[ppg_column]
    df_view["ppg_delta_ppr_minus_half"] = (
        df_view["fantasy_points_per_game_ppr"] - df_view["fantasy_points_per_game_half_ppr"]
    )
    display_columns = [
        "player",
        "position",
        "season",
        "adp_overall",
        "games_played",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "touchdowns",
        "fantasy_points",
        "fantasy_points_per_game",
        "ppg_delta_ppr_minus_half",
    ]

    tab_data, tab_charts, tab_advisor = st.tabs(["Data", "Charts", "Draft Advisor"])

    with tab_data:
        st.subheader("Sample Player Data")
        st.caption(f"Scoring format: {scoring_format.upper()}")
        st.write("Using a small, local CSV to power the demo visuals.")
        st.dataframe(df_view[display_columns], width="stretch")
        st.markdown(
            """
            **Notes**
            - `adp_overall` is used to compute draft rounds for a 12-team league.
            - Fantasy points are computed with `src/scoring.py`.
            - Values are illustrative and intended for demo purposes.
            """
        )

    with tab_charts:
        st.subheader("Charts")
        st.plotly_chart(
            plot_value_by_round(df_view, "fantasy_points_per_game").update_layout(
                title=f"Value by Draft Round (Avg PPG) - {scoring_format.upper()}"
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_adp_vs_fppg(df_view, "fantasy_points_per_game").update_layout(
                title=f"ADP vs PPG - {scoring_format.upper()}"
            ),
            width="stretch",
        )

    with tab_advisor:
        st.subheader("Draft Advisor Summary")
        summary = (
            df_view.groupby(["draft_round", "position"], as_index=False)["fantasy_points_per_game"]
            .mean()
            .rename(columns={"fantasy_points_per_game": "avg_ppg"})
        )
        top_values = summary.sort_values("avg_ppg", ascending=False).head(3)
        top_bullets = [
            f"Round {int(row.draft_round)} {row.position} (~{row.avg_ppg:.1f} PPG)"
            for row in top_values.itertuples(index=False)
        ]
        st.write(
            f"""
            Scoring format: **{scoring_format.upper()}**.

            Based on the computed results, the strongest value pockets by average points per game are:
            - {top_bullets[0] if len(top_bullets) > 0 else "No data"}
            - {top_bullets[1] if len(top_bullets) > 1 else "No data"}
            - {top_bullets[2] if len(top_bullets) > 2 else "No data"}

            Use these as a guide: prioritize early-round players at positions showing the highest
            PPG in your selected scoring format, and look for later-round positions where ADP lags
            behind per-game output.
            """
        )

        st.markdown("---")
        st.markdown("**Gemini-Powered Explanation (Coming Soon)**")
        st.info("Placeholder: Gemini-generated draft strategy explanation will appear here.")
        st.code(
            """
# TODO: Call Gemini API here
# - Send chart summaries and roster context
# - Receive structured recommendations
# - Render bullet-point insights
            """.strip()
        )


if __name__ == "__main__":
    main()
