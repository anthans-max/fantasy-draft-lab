import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import pandas as pd
import plotly.express as px
import streamlit as st

from src.scoring import calculate_fantasy_points
from src.vor import DEFAULT_REPLACEMENT_RANKS, add_draft_round, add_vor


st.set_page_config(page_title="Fantasy Draft Analytics Demo", layout="wide")

DATA_PATH = REPO_ROOT / "data" / "sample_players.csv"
LEAGUE_TEAMS = 12

@st.cache_data(show_spinner="Loading fantasy dataset…")
def load_fantasy_data() -> pd.DataFrame:
    """
    Loads fantasy data locally when available, otherwise uses a configured remote URL.
    """
    repo_root = Path(__file__).resolve().parents[1]
    local_parquet = repo_root / "data" / "fantasy_players_2019_2025.parquet"
    local_csv = repo_root / "data" / "sample_players.csv"

    if local_parquet.exists() and local_parquet.stat().st_size > 0:
        return pd.read_parquet(local_parquet)

    if local_csv.exists() and local_csv.stat().st_size > 0:
        return pd.read_csv(local_csv)

    url = os.getenv("FANTASY_DATA_URL", "").strip()
    fmt = os.getenv("FANTASY_DATA_FORMAT", "parquet").strip().lower()

    if url:
        if fmt == "parquet":
            return pd.read_parquet(url)
        if fmt == "csv":
            return pd.read_csv(url)
        raise ValueError(f"Unsupported FANTASY_DATA_FORMAT: {fmt}")

    raise RuntimeError(
        "No local data files found at data/fantasy_players_2019_2025.parquet "
        "or data/sample_players.csv, and FANTASY_DATA_URL is not set."
    )



@st.cache_data
def load_players() -> pd.DataFrame:
    df = load_fantasy_data()
    return add_draft_round(
        df,
        league_teams=LEAGUE_TEAMS,
        ppg_column="fantasy_points_per_game_ppr",
    )



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


def build_player_summary(df_filtered: pd.DataFrame, scoring_mode: str) -> pd.DataFrame:
    if df_filtered.empty:
        return df_filtered.copy()

    mode = (scoring_mode or "").lower()
    use_ppr = "ppr" in mode and "half" not in mode
    points_column = "fantasy_points_ppr" if use_ppr else "fantasy_points_half_ppr"
    ppg_column = (
        "fantasy_points_per_game_ppr"
        if use_ppr
        else "fantasy_points_per_game_half_ppr"
    )

    adp_candidates = ["adp_overall", "adp overall", "adp", "adp_pick", "adp_rank", "adp_overall_rank"]
    adp_column = next((col for col in adp_candidates if col in df_filtered.columns), None)

    agg_map = {
        "season": "nunique",
        ppg_column: "mean",
        points_column: "mean",
        "games_played": "mean",
        "touchdowns": "mean",
        "rushing_yards": "mean",
        "receiving_yards": "mean",
        "receptions": "mean",
    }
    if adp_column:
        agg_map[adp_column] = "mean"

    summary = (
        df_filtered.groupby(["player", "position"], as_index=False)
        .agg(agg_map)
        .rename(
            columns={
                "season": "seasons_played",
                ppg_column: "avg_ppg",
                points_column: "avg_fantasy_points",
                "games_played": "avg_games",
                "touchdowns": "avg_touchdowns",
                "rushing_yards": "avg_rushing_yards",
                "receiving_yards": "avg_receiving_yards",
                "receptions": "avg_receptions",
            }
        )
    )
    if adp_column:
        summary = summary.rename(columns={adp_column: "avg_adp"})

    return summary.sort_values("avg_ppg", ascending=False)


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
    min_season = int(df["season"].min())
    max_season = int(df["season"].max())

    st.sidebar.header("Settings")
    scoring_choice = st.sidebar.radio(
        "Scoring Format", ["Half-PPR", "Full PPR"], index=0
    )
    if min_season == max_season:
        st.sidebar.warning("Only one season found in data; season filter disabled.")
        season_selection = st.sidebar.selectbox(
            "Season", [min_season], index=0, disabled=True
        )
        season_filter = season_selection
    else:
        season_range = st.sidebar.slider(
            "Seasons",
            min_value=min_season,
            max_value=max_season,
            value=(min_season, max_season),
        )
        season_filter = season_range

    st.sidebar.subheader("Positions")
    position_flags = {
        "QB": st.sidebar.checkbox("QB", value=True),
        "RB": st.sidebar.checkbox("RB", value=True),
        "WR": st.sidebar.checkbox("WR", value=True),
        "TE": st.sidebar.checkbox("TE", value=True),
    }
    selected_positions = [pos for pos, checked in position_flags.items() if checked]

    use_full_ppr = scoring_choice == "Full PPR"
    scoring_label = "Full PPR" if use_full_ppr else "Half-PPR"
    position_label = ", ".join(selected_positions) if selected_positions else "None"
    if isinstance(season_filter, tuple):
        season_label = f"{season_filter[0]}–{season_filter[1]}"
    else:
        season_label = f"{season_filter}"
    st.markdown(
        f"**Current Settings:** {scoring_label} | Seasons {season_label} | "
        f"Positions: {position_label}"
    )

    if not selected_positions:
        st.warning("Select at least one position in Settings to view data and charts.")
        return

    points_column = "fantasy_points_ppr" if use_full_ppr else "fantasy_points_half_ppr"
    ppg_column = (
        "fantasy_points_per_game_ppr"
        if use_full_ppr
        else "fantasy_points_per_game_half_ppr"
    )

    if isinstance(season_filter, tuple):
        season_mask = (df["season"] >= season_filter[0]) & (df["season"] <= season_filter[1])
    else:
        season_mask = df["season"] == season_filter

    df_filtered = df[season_mask & (df["position"].isin(selected_positions))].copy()

    df_view = df_filtered.copy()
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

    tab_data, tab_charts, tab_vor, tab_advisor = st.tabs(
        ["Data", "Charts", "VOR by Draft Round", "Draft Advisor"]
    )

    with tab_data:
        st.subheader("Sample Player Data")
        st.caption(f"Scoring format: {scoring_label}")
        st.write("Using a small, local CSV to power the demo visuals.")
        table_view = st.radio(
            "Table view",
            ["Player Averages", "Raw Season Data"],
            index=0,
            horizontal=True,
        )

        max_seasons_played = (
            int(df_view.groupby(["player", "position"])["season"].nunique().max())
            if not df_view.empty
            else 1
        )
        min_seasons_played = st.number_input(
            "Min seasons played",
            min_value=1,
            max_value=max_seasons_played,
            value=2,
            step=1,
            disabled=table_view == "Raw Season Data",
        )

        if table_view == "Player Averages":
            summary = build_player_summary(df_view, scoring_choice)
            if min_seasons_played > 1:
                summary = summary[summary["seasons_played"] >= min_seasons_played]

            if summary.empty:
                st.warning("No players match the current filters.")
            else:
                display = summary.rename(
                    columns={
                        "player": "Player",
                        "position": "Pos",
                        "seasons_played": "Seasons Played",
                        "avg_ppg": "Avg PPG",
                        "avg_fantasy_points": "Avg Fantasy Points",
                        "avg_games": "Avg Games",
                        "avg_touchdowns": "Avg TDs",
                        "avg_rushing_yards": "Avg Rush Yds",
                        "avg_receiving_yards": "Avg Rec Yds",
                        "avg_receptions": "Avg Receptions",
                        "avg_adp": "Avg ADP",
                    }
                )
                round_map = {
                    "Avg PPG": 2,
                    "Avg Fantasy Points": 1,
                    "Avg Games": 1,
                    "Avg TDs": 1,
                    "Avg Rush Yds": 0,
                    "Avg Rec Yds": 0,
                    "Avg Receptions": 1,
                    "Avg ADP": 1,
                }
                display = display.round({k: v for k, v in round_map.items() if k in display.columns})
                if "Seasons Played" in display.columns:
                    display["Seasons Played"] = display["Seasons Played"].astype(int)
                st.dataframe(display, width="stretch")
        else:
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
            plot_value_by_round(df_view, ppg_column).update_layout(
                title=f"Value by Draft Round (Avg PPG) - {scoring_label}"
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_adp_vs_fppg(df_view, ppg_column).update_layout(
                title=f"ADP vs PPG - {scoring_label}"
            ),
            width="stretch",
        )

    with tab_vor:
        st.subheader("VOR by Draft Round")
        st.caption(
            "Value over replacement is computed per position and season using the selected "
            "scoring format."
        )

        df_vor = add_vor(df_filtered, ppg_column=ppg_column, replacement_ranks=DEFAULT_REPLACEMENT_RANKS)

        if df_vor.empty:
            st.warning("No data available for the current filters.")
        else:
            view_choice = st.radio(
                "Chart view",
                ["Average VOR", "Total VOR"],
                index=0,
                horizontal=True,
            )
            display_metric = "avg_vor" if view_choice == "Average VOR" else "total_vor"
            display_label = "Average VOR (PPG)" if view_choice == "Average VOR" else "Total VOR (PPG)"

            round_order = [str(i) for i in range(1, 21)] + ["20+"]
            if "Unknown" in df_vor["draft_round_bucket"].unique():
                round_order.append("Unknown")

            df_vor["draft_round_bucket"] = pd.Categorical(
                df_vor["draft_round_bucket"], categories=round_order, ordered=True
            )

            summary = (
                df_vor.groupby(["draft_round_bucket", "position"], as_index=False)["vor"]
                .agg(avg_vor="mean", total_vor="sum")
                .rename(columns={"draft_round_bucket": "draft_round"})
            )
            summary["draft_round"] = pd.Categorical(
                summary["draft_round"], categories=round_order, ordered=True
            )
            summary = summary.sort_values("draft_round")

            fig = px.bar(
                summary,
                x="draft_round",
                y=display_metric,
                color="position",
                barmode="group",
                title=f"{view_choice} by Draft Round - {scoring_label}",
                labels={
                    "draft_round": "Draft Round",
                    display_metric: display_label,
                },
            )
            st.plotly_chart(fig, width="stretch")
            st.dataframe(summary, width="stretch")

            total_fig = px.bar(
                summary,
                x="draft_round",
                y="total_vor",
                color="position",
                barmode="stack",
                title=f"Total VOR by Draft Round (Stacked) - {scoring_label}",
                labels={
                    "draft_round": "Draft Round",
                    "total_vor": "Total VOR (PPG)",
                },
            )
            st.plotly_chart(total_fig, width="stretch")

            if (df_vor["adp_source"] == "Demo ADP").any():
                st.info("Draft rounds are based on Demo ADP (synthetic rank-based values).")

            detail = (
                df_vor.sort_values(["draft_round_bucket", "vor"], ascending=[True, False])
                .groupby("draft_round_bucket")
                .head(5)
            )
            detail_view = detail[
                [
                    "player",
                    "season",
                    "position",
                    ppg_column,
                    "replacement_ppg",
                    "vor",
                    "draft_round",
                    "draft_pick_used",
                    "adp_used",
                    "adp_source",
                ]
            ].rename(
                columns={
                    "player": "player_name",
                    ppg_column: "ppg",
                    "replacement_ppg": "replacement_ppg",
                    "vor": "vor",
                    "draft_round": "draft_round",
                    "draft_pick_used": "draft_pick",
                    "adp_used": "adp",
                    "adp_source": "adp_source",
                }
            )
            st.subheader("Top Players by VOR (Top 5 per Round)")
            st.dataframe(detail_view, width="stretch")

    with tab_advisor:
        st.subheader("Draft Advisor Summary")
        summary = (
            df_view.groupby(["draft_round", "position"], as_index=False)[ppg_column]
            .mean()
            .rename(columns={ppg_column: "avg_ppg"})
        )
        top_values = summary.sort_values("avg_ppg", ascending=False).head(3)
        top_bullets = [
            f"Round {int(row.draft_round)} {row.position} (~{row.avg_ppg:.1f} PPG)"
            for row in top_values.itertuples(index=False)
        ]
        st.write(
            f"""
            Scoring format: **{scoring_label}**.

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
