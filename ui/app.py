import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Fantasy Draft Analytics Demo", layout="wide")

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_players.csv"
LEAGUE_TEAMS = 12


@st.cache_data
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["draft_round"] = df["adp_overall"].apply(lambda adp: math.ceil(adp / LEAGUE_TEAMS))
    return df


def plot_value_by_round(df: pd.DataFrame):
    summary = (
        df.groupby(["draft_round", "position"], as_index=False)["fantasy_points_per_game"]
        .mean()
        .rename(columns={"fantasy_points_per_game": "avg_fppg"})
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


def plot_adp_vs_fppg(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="adp_overall",
        y="fantasy_points_per_game",
        color="position",
        hover_name="player",
        title="ADP vs Fantasy Points Per Game",
        labels={"adp_overall": "ADP (Overall)", "fantasy_points_per_game": "Fantasy Points/Game"},
    )
    fig.update_layout(xaxis=dict(autorange="reversed"))
    return fig


def main():
    st.title("Fantasy Football Draft Analytics")
    st.caption("Demo app for draft value exploration (local CSV data).")

    df = load_players()

    tab_data, tab_charts, tab_advisor = st.tabs(["Data", "Charts", "Draft Advisor"])

    with tab_data:
        st.subheader("Sample Player Data")
        st.write("Using a small, local CSV to power the demo visuals.")
        st.dataframe(df, width="stretch")
        st.markdown(
            """
            **Notes**
            - `adp_overall` is used to compute draft rounds for a 12-team league.
            - Values are illustrative and intended for demo purposes.
            """
        )

    with tab_charts:
        st.subheader("Charts")
        st.plotly_chart(plot_value_by_round(df), width="stretch")
        st.plotly_chart(plot_adp_vs_fppg(df), width="stretch")

    with tab_advisor:
        st.subheader("Draft Advisor Summary")
        st.write(
            """
            The charts suggest that early rounds tend to be dominated by elite RB/WR production,
            while certain positions (notably QB/TE) can offer strong per-game value in later rounds.
            In a balanced build, prioritize high-ceiling RB/WR in the first few rounds, then target
            value pockets where ADP lags behind per-game output.
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
